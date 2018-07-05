#include "training/graph_group_sync.h"
#include "functional/functional.h"
#include "tensors/tensor_operators.h"

namespace marian {

SyncGraphGroup::SyncGraphGroup(Ptr<Config> config)
    : GraphGroup(config),
      devices_{options_->getDevices()},
      movingAvg_{options_->get<float>("exponential-smoothing") > 0},
      mvDecay_{options_->get<float>("exponential-smoothing")},
      delay_{options_->get<size_t>("optimizer-delay")} {

  for(auto device : devices_) {
    auto graph = New<ExpressionGraph>();
    graph->setDevice(device);
    graph->reserveWorkspaceMB(options_->get<size_t>("workspace"));
    graph->getBackend()->setClip(options_->get<float>("clip-gemm"));

    graphs_.push_back(graph);
    shardOpt_.push_back(Optimizer(options_));
    builders_.push_back(models::from_config(options_, models::usage::training));
  }

  comm_ = createCommunicator(graphs_, options_->get<bool>("no-nccl", false));
}

void SyncGraphGroup::setScheduler(Ptr<Scheduler> scheduler) {
  scheduler_ = scheduler;
  // optimizer has to be registered last to see changes of learning rate
  scheduler_->registerTrainingObserver(scheduler_);

  for(auto opt : shardOpt_)
    scheduler_->registerTrainingObserver(opt);
}

void SyncGraphGroup::updateMovingAverage(Tensor paramsAvg,
                                         Tensor params,
                                         size_t batches) {
  using namespace functional;
  float decay
      = std::max(mvDecay_, 1.f - (float)(batches + 1) / (float)(batches + 10));
  Element(_1 = ((1.f - decay) * _1) + (decay * _2), paramsAvg, params);
}

void SyncGraphGroup::initialize(const std::vector<Ptr<data::Batch>>& batches) {
  // Initialize 0th graph with random weights in one forward step
  {
    THREAD_GUARD(builders_[0]->build(graphs_[0], batches[0]);
                 graphs_[0]->forward(););

    // Copy weights from 0th graph to all other graphs
    // to have equal weights across devices
    ThreadPool pool(graphs_.size() - 1, graphs_.size() - 1);
    for(size_t i = 1; i < graphs_.size(); ++i) {
      auto init = [&](size_t i) {
        // initialize i-th graph and weights
        builders_[i]->build(graphs_[i], batches[0]);
        graphs_[i]->forward();
        // overwrite weights of i-th graph with weights from 0th graph
        graphs_[i]->params()->vals()->copyFrom(graphs_[0]->params()->vals());
      };
      pool.enqueue(init, i);
    }
  }

  if(movingAvg_ && paramsAvg_.size() == 0) {
    int totalSize = graphs_[0]->params()->vals()->size();
    shardSize_ = ceil(totalSize / (float)devices_.size());

    int pos = 0;
    for(auto graph : graphs_) {
      int __size__ = std::min(shardSize_, totalSize);

      auto paramsAlloc = New<TensorAllocator>(graph->getBackend());
      paramsAllocs_.push_back(paramsAlloc);

      paramsAlloc->reserveExact(__size__ * sizeof(float));

      Tensor paramAvg;
      paramsAlloc->allocate(paramAvg, {1, __size__});
      paramsAvg_.push_back(paramAvg);

      paramAvg->copyFrom(graphs_[0]->params()->vals()->subtensor(pos, __size__));

      // move to next shard
      pos += __size__;
      totalSize -= __size__;
    }
  }
}

void SyncGraphGroup::execute(Ptr<data::Batch> batch) {
  size_t devs = devices_.size();
  auto batches = batch->split(delay_ * devs);

  float div = batches.size();  // no. of batches
  // do not average gradients if cost type is sum.
  if(options_->get<std::string>("cost-type") == "ce-sum")
    div = 1;

  std::vector<std::vector<Ptr<data::Batch>>> delayedBatches;

  for(int i = 0; i < delay_; ++i) {
    if(i * devs < batches.size()) {
      delayedBatches.emplace_back();
      for(int j = 0; j < devs; ++j) {
        size_t index = i * devs + j;
        if(index < batches.size())
          delayedBatches.back().push_back(batches[i * devs + j]);
        else
          delayedBatches.back().push_back(nullptr);
      }
    }
  }

  std::vector<float> costs(devices_.size(), 0.f);
  size_t t = 1;

  for(const auto& curBatches : delayedBatches) {
    if(first_) {
      initialize(curBatches);
      first_ = false;
    }

    // Execute single forward/backward step
    auto forwardBackward = [this, &costs, curBatches, t](size_t idx, int pos) {
      auto graph = graphs_[idx];
      auto batch = curBatches[idx];

      if(batch) {
        auto costNode = builders_[idx]->build(graph, batch);
        graph->forward();
        costs[idx] += costNode->scalar();

        // only reset gradients to 0 if t == 1
        graph->backward(t == 1);
      }
      else {
        // handle case of empty batch, execute do-nothing fw-bw step for
        // proper inits and resets.
        graph->forward();
        // only reset gradients to 0 if t == 1
        graph->backward(t == 1);
      }
    };

    // Update parameter shard with gradient shard
    auto update = [this, div](size_t idx, int pos) {
      int totalSize = graphs_[0]->params()->vals()->size();
      int shardSize = ceil(totalSize / (float)devices_.size());

      int size = std::min(totalSize - pos, shardSize);

      auto curGrad  = graphs_[idx]->params()->grads()->subtensor(pos, size);
      auto curParam = graphs_[idx]->params()->vals()->subtensor(pos, size);

      if(div != -1) {
        using namespace functional;
        Element(_1 = _1 / div, curGrad);
      }

      shardOpt_[idx]->update(curParam, curGrad);

      if(movingAvg_)
        updateMovingAverage(paramsAvg_[idx], curParam, scheduler_->numberOfBatches());
    };

    comm_->foreach(forwardBackward);
    if(t == delayedBatches.size()) {
      comm_->scatterReduce();
      comm_->foreach(update);
      comm_->allGather();
    }

    t++;
  }

  float cost = 0;
  for(auto& c : costs) {
    cost += c;
    c = 0;
  }

  // @TODO: review this
  if(options_->get<std::string>("cost-type") != "ce-sum") {
    cost = cost / (costs.size() * delay_);
  }

  if(scheduler_) {
    scheduler_->update(cost, batches);

    if(scheduler_->saving()) {
      this->save();
    }

    if(scheduler_->validating()) {
      if(movingAvg_) {
        comm_->swapParams(paramsAvg_);
      }

      // safe, because all graphs are idle during validation with sync sgd
      scheduler_->validate(graphs_);

      if(movingAvg_) {
        comm_->swapParams(paramsAvg_);
      }
    }
  }
}

void SyncGraphGroup::load() {
  if(!options_->get<bool>("no-reload")) {
    std::string name = options_->get<std::string>("model");

    if(boost::filesystem::exists(name)) {
      size_t i = 0;
      if(scheduler_)
        scheduler_->load(name);
      for(auto graph : graphs_)
        builders_[i++]->load(graph, name);

      // @TODO: probably we want to have the list of DeviceIds as an attribute
      std::vector<Ptr<Backend>> backends;
      for(auto graph : graphs_)
        backends.push_back(graph->getBackend());
      shardOpt_[0]->load(name + ".optimizer.npz", shardOpt_, backends);

    } else if(options_->has("pretrained-model")) {
      std::string init = options_->get<std::string>("pretrained-model");
      LOG(info,
          "Initialize model weights with the pre-trained model {}",
          init);
      size_t i = 0;
      for(auto graph : graphs_)
        builders_[i++]->load(graph, init, false);
    }
  }
}

void SyncGraphGroup::save(bool final) {
    if(final && scheduler_) {
      if(movingAvg_ && paramsAvg_.size() > 0)
        comm_->swapParams(paramsAvg_);

      scheduler_->validate(graphs_, true);

      if(movingAvg_ && paramsAvg_.size() > 0)
        comm_->swapParams(paramsAvg_);
    }
    save(graphs_[0], final);
  }

  void SyncGraphGroup::save(Ptr<ExpressionGraph> graph, bool final) {
    int idx = 0;
    for(int i = 0; i < graphs_.size(); ++i) {
      if(graph == graphs_[i]) {
        idx = i;
        break;
      }
    }

    if(movingAvg_ && paramsAvg_.size() > 0)
      comm_->swapParams(paramsAvg_);

    std::string name = options_->get<std::string>("model");

    if(options_->get<bool>("overwrite")) {
      builders_[idx]->save(graphs_[idx], name, true);
      if(scheduler_)
        scheduler_->save(name);
    } else {
      if(!final) {
        std::string numberOfBatches
            = scheduler_ ? std::to_string(scheduler_->numberOfBatches())
                         : "unknown";
        std::string nameOverwrite = name;
        nameOverwrite.replace(
            name.size() - 4, 4, ".iter" + numberOfBatches + ".npz");
        builders_[idx]->save(graphs_[idx], nameOverwrite);
      }

      builders_[idx]->save(graphs_[idx], name, true);
      if(scheduler_)
        scheduler_->save(name);
    }

    if(movingAvg_ && paramsAvg_.size() > 0)
      comm_->swapParams(paramsAvg_);

    size_t totalSize = graphs_[idx]->params()->vals()->size();
    shardOpt_[idx]->save(name + ".optimizer.npz", shardOpt_, totalSize);
  }

}
