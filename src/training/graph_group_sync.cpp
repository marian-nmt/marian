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

  comm_ = createCommunicator(graphs_);
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

void SyncGraphGroup::fetchParams(Tensor oldParams,
                                 const std::vector<Tensor>& params) {
  // @TODO read guard on parameters
  int pos = 0;
  std::vector<std::thread> threads;
  for(int idx = 0; idx < devices_.size(); idx++) {
    threads.emplace_back(std::thread(
        [=](int idx, int pos) {
          oldParams->subtensor(pos, params[idx]->size())->copyFrom(params[idx]);
        },
        idx,
        pos));
    pos += shardSize_;
  }
  for(auto&& t : threads) {
    t.join();
  }
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
      paramAvg->copyFrom(graphs_[0]->params()->vals()->subtensor(pos, __size__));
      paramsAvg_.push_back(paramAvg);

      // move to next shard
      pos += __size__;
      totalSize -= __size__;
    }
  }
}

void SyncGraphGroup::execute(const std::vector<Ptr<data::Batch>>& batches) {
  // if there are fewer batches than we need, split last batch into right number
  // of pieces and replace last batch with the splits.

  float div = batches.size();  // no. of batches
  // do not average gradients if cost type is sum.
  if(options_->get<std::string>("cost-type") == "ce-sum")
    div = 1;

  std::vector<std::vector<Ptr<data::Batch>>> delayedBatches;
  size_t devs = devices_.size();

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
    };

    // Update parameter shard with gradient shard
    auto update = [this](size_t idx, int pos) {
      int totalSize = graphs_[0]->params()->vals()->size();
      int shardSize = ceil(totalSize / (float)devices_.size());

      int size = std::min(totalSize - pos, shardSize);

      auto curGrad  = graphs_[idx]->params()->grads()->subtensor(pos, size);
      auto curParam = graphs_[idx]->params()->vals()->subtensor(pos, size);
      shardOpt_[idx]->update(curParam, curGrad);

      if(movingAvg_)
        updateMovingAverage(
          paramsAvg_[idx], curParam, scheduler_->numberOfBatches());
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
      if(movingAvg_)
        for(auto graph : graphs_)
          fetchParams(graph->params()->vals(), paramsAvg_);

      // safe, because all graphs are idle during validation with sync sgd
      scheduler_->validate(graphs_);

      //if(movingAvg_)
      //  for(auto graph : graphs_)
      //    fetchParams(graph->params()->vals(), params_);
    }
  }
}
}
