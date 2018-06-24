#include "training/graph_group_sync.h"
#include "functional/functional.h"
#include "tensors/tensor_operators.h"

namespace marian {

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

void SyncGraphGroup::foreachDevice(const std::function<void(size_t, int)>& task) {
  std::vector<std::thread> group;
  int pos = 0;
  for(int idx = 0; idx < devices_.size(); ++idx) {
    group.emplace_back(task, idx, pos);
    pos += params_[idx]->size();
  }
  for(auto& t : group)
    t.join();
}

void SyncGraphGroup::execute(const std::vector<Ptr<data::Batch>>& batches) {
  // if there are fewer batches than we need, split last batch into right number
  // of pieces and replace last batch with the splits.

  std::vector<Ptr<data::Batch>> newBatches = batches;
  if(newBatches.size() < numBatches()) {
    size_t splitFill = numBatches() - newBatches.size() + 1;
    auto fillerBatches = newBatches.back()->split(splitFill);
    newBatches.back() = fillerBatches[0];
    for(int i = 1; i < splitFill; ++i)
      newBatches.push_back(fillerBatches[i]);
  }

  std::vector<std::vector<Ptr<data::Batch>>> delayedBatches;
  for(int i = 0; i < delay_; ++i) {
    delayedBatches.emplace_back();
    size_t devs = devices_.size();
    for(int j = 0; j < devs; ++j) {
      delayedBatches.back().push_back(newBatches[i * devs + j]);
    }
  }

  std::vector<float> costs(devices_.size(), 0.f);
  size_t t = 1;

  for(const auto& batches : delayedBatches) {
    if(first_) {
      {
        // Initialize 0th graph with random weights in one forward step
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

      // Initialize sharded parameter storage. For n devices
      // each device stores 1/n-th of parameters.
      // We also create sharded gradients and temporary storage.
      if(params_.size() == 0) {
        int totalSize = graphs_[0]->params()->vals()->size();
        shardSize_ = ceil(totalSize / (float)devices_.size());

        int pos = 0;
        for(auto graph : graphs_) {
          int __size__ = std::min(shardSize_, totalSize);

          auto paramsAlloc = New<TensorAllocator>(graph->getBackend());
          paramsAllocs_.push_back(paramsAlloc);

          size_t chunks = movingAvg_ ? 3 : 2;
          paramsAlloc->reserveExact(chunks * __size__ * sizeof(float));

          Tensor param, tmp, paramAvg;

          // set parameters to actual value from 0th graph
          paramsAlloc->allocate(param, {1, __size__});
          params_.push_back(param);
          param->copyFrom(graphs_[0]->params()->vals()->subtensor(pos, __size__));

          paramsAlloc->allocate(tmp, {1, __size__});
          tmpTensors_.push_back(tmp);

          if(movingAvg_) {
            paramsAlloc->allocate(paramAvg, {1, __size__});
            paramAvg->copyFrom(param);
            paramsAvg_.push_back(paramAvg);
          }

          // move to next shard
          pos += __size__;
          totalSize -= __size__;
        }
      }
      first_ = false;
    }

    // execute single forward/backward step
    auto taskForwardBackward = [this, &costs, batches, t](size_t idx, int pos) {
      auto graph = graphs_[idx];
      auto batch = batches[idx];

      auto costNode = builders_[idx]->build(graph, batch);
      graph->forward();
      costs[idx] += costNode->scalar();
      graph->backward(t == 1);
    };

    // device index corresponds to shard index
    auto taskGather = [this, batches](size_t idx, int pos) {
      int shardSize = params_[idx]->size();

      float div = devices_.size();  // no. of GPUs
      // do not average gradients if cost type is sum.
      if(options_->get<std::string>("cost-type") == "ce-sum")
        div = 1;

      auto curGrad = graphs_[idx]->params()->grads()->subtensor(pos, shardSize);

      // collect and sum gradients
      // to be replaced with ncclScatterReduce
      for(auto graph : graphs_) {
        if(graph != graphs_[idx]) {
          auto subGrad = graph->params()->grads()->subtensor(pos, shardSize);
          tmpTensors_[idx]->copyFrom(subGrad);

          using namespace functional;
          Element(_1 = _1 + (_2 / div), curGrad, tmpTensors_[idx]);
        }
      }
    };

    auto taskUpdate = [this](size_t idx, int pos) {
      int shardSize = params_[idx]->size();
      auto curGrad = graphs_[idx]->params()->grads()->subtensor(pos, shardSize);
      shardOpt_[idx]->update(params_[idx], curGrad);

      if(movingAvg_)
        updateMovingAverage(
          paramsAvg_[idx], params_[idx], scheduler_->numberOfBatches());
    };

    auto taskBroadcast = [this](size_t idx, int pos) {
      int shardSize = params_[idx]->size();
      auto curGrad = graphs_[idx]->params()->grads()->subtensor(pos, shardSize);

      // copy parameter shard to each graph
      for(auto graph : graphs_) {
        auto subParam = graph->params()->vals()->subtensor(pos, shardSize);
        subParam->copyFrom(params_[idx]);
      }
    };

    foreachDevice(taskForwardBackward);

    if(t == delay_) {
      foreachDevice(taskGather);
      foreachDevice(taskUpdate);
      foreachDevice(taskBroadcast);
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

      if(movingAvg_)
        for(auto graph : graphs_)
          fetchParams(graph->params()->vals(), params_);
    }
  }
}
}
