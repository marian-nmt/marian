#include "kernels/tensor_operators.h"
#include "training/graph_group_sync.h"

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
  float decay = std::max(mvDecay_, 1.f - (float)(batches + 1) / (float)(batches + 10));
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

void SyncGraphGroup::execute(Ptr<data::Batch> batch) {
  std::vector<Ptr<data::Batch>> batches = batch->split(devices_.size());

  if(first_) {
    for(size_t i = 0; i < graphs_.size(); ++i) {
      // takes care of thead_local stuff
      THREAD_GUARD(builders_[i]->build(graphs_[i], batches[0]);
                   graphs_[i]->forward(););

      if(i > 0)
        graphs_[i]->params()->vals()->copyFrom(graphs_[0]->params()->vals());
    }

    if(params_.size() == 0) {
      int totalSize = graphs_[0]->params()->vals()->size();
      shardSize_ = ceil(totalSize / (float)devices_.size());

      int pos = 0;
      for(auto graph : graphs_) {
        int __size__ = min(shardSize_, totalSize);

        auto paramsAlloc = New<TensorAllocator>(graph->getBackend());
        paramsAllocs_.push_back(paramsAlloc);

        paramsAlloc->reserveExact(3 * __size__ * sizeof(float));

        Tensor param, grad, tmp;
        paramsAlloc->allocate(param, {1, __size__});
        paramsAlloc->allocate(grad, {1, __size__});
        paramsAlloc->allocate(tmp, {1, __size__});
        params_.push_back(param);
        grads_.push_back(grad);
        tmpTensors_.push_back(tmp);

        param->copyFrom(graphs_[0]->params()->vals()->subtensor(pos, __size__));
        pos += __size__;
        totalSize -= __size__;
      }
    }

    if(movingAvg_ && paramsAvg_.size() == 0) {
      int totalSize = graphs_[0]->params()->vals()->size();

      int i = 0;
      for(auto graph : graphs_) {
        int __size__ = min(shardSize_, totalSize);
        totalSize -= __size__;
        Tensor paramAvg;
        auto allocator = New<TensorAllocator>(graph->getBackend());

        allocator->reserveExact(__size__ * sizeof(float));
        allocator->allocate(paramAvg, {1, __size__});

        paramAvg->copyFrom(params_[i++]);

        paramsAllocAvg_.push_back(allocator);
        paramsAvg_.push_back(paramAvg);
      }
    }

    first_ = false;
  }

  std::vector<float> costs(devices_.size());

  {
    auto task = [this, &costs, batches](size_t idx) {
      auto graph = graphs_[idx];
      auto batch = batches[idx];

      if(batch->size() > 0) {
        auto costNode = builders_[idx]->build(graph, batch);
        graph->forward();
        costs[idx] = costNode->scalar();
        graph->backward();
      }
    };

    ThreadPool pool(devices_.size(), devices_.size());
    for(int idx = 0; idx < batches.size(); ++idx)
      pool.enqueue(task, idx);
  }

  {
    auto task = [this, batches](size_t idx, int pos) {
      grads_[idx]->set(0);
      int size = params_[idx]->size();
      int i = 0;
      float div = devices_.size(); // no. of GPUs
      for(auto graph : graphs_) {
        if(batches[i]->size() > 0) {
          auto subGrad = graph->params()->grads()->subtensor(pos, size);
          tmpTensors_[idx]->copyFrom(subGrad);

          using namespace functional;
          Element(_1 = _1 + (_2 / div), grads_[idx], tmpTensors_[idx]);
        }
        i++;
      }

      shardOpt_[idx]->update(params_[idx], grads_[idx]);

      if(movingAvg_)
        updateMovingAverage(
            paramsAvg_[idx], params_[idx], scheduler_->numberOfBatches());

      for(auto graph : graphs_) {
        auto subParam = graph->params()->vals()->subtensor(pos, size);
        subParam->copyFrom(params_[idx]);
      }
    };

    ThreadPool pool(devices_.size(), devices_.size());
    int pos = 0;
    for(int idx = 0; idx < devices_.size(); ++idx) {
      pool.enqueue(task, idx, pos);
      pos += params_[idx]->size();
    }
  }

  float cost = 0;
  for(auto c : costs)
    cost += c;
  cost = cost / costs.size();

  if(scheduler_) {
    scheduler_->update(cost, batch);

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
