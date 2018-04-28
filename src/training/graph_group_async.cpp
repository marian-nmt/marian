#include "training/graph_group_async.h"
#include "functional/functional.h"
#include "tensors/tensor_operators.h"
#include "data/corpus_base.h"

namespace marian {

void AsyncGraphGroup::setScheduler(Ptr<Scheduler> scheduler) {
  scheduler_ = scheduler;
  // optimizer has to be registered last to see changes of learning rate
  scheduler_->registerTrainingObserver(scheduler_);

  for(auto opt : shardOpt_)
    scheduler_->registerTrainingObserver(opt);
}

void AsyncGraphGroup::fetchParams(Tensor oldParams,
                                  const std::vector<Tensor>& params,
                                  int device_id) {
  // @TODO read guard on parameters
  int pos = 0;

  std::vector<std::thread> threads;
  for(int idx = 0; idx < devices_.size(); idx++) {
    threads.emplace_back(std::thread(
        [&](int idx, int pos) {
          // individual mutex per-shard
          std::lock_guard<std::mutex> guard(shardSync_[idx]);
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

void AsyncGraphGroup::pushGradients(Tensor newGrads,
                                    size_t batch_words,
                                    int device_id) {
  // add instead of copy?
  std::vector<std::thread> threads;
  int pos = 0;
  for(int idx = 0; idx < devices_.size(); idx++) {
    threads.emplace_back(std::thread(
        [&](int idx, int pos) {
          // individual mutex per-shard
          std::lock_guard<std::mutex> guard(shardSync_[idx]);
          grads_[idx]->copyFrom(newGrads->subtensor(pos, grads_[idx]->size()));

          if(scaleLearningRate_) {
            shardOpt_[idx]->update(
                params_[idx], grads_[idx], batch_words / avgBatchWords_);
          } else {
            shardOpt_[idx]->update(params_[idx], grads_[idx]);
          }

          if(movingAvg_)
            updateMovingAverage(
                paramsAvg_[idx], params_[idx], scheduler_->numberOfBatches());
        },
        idx,
        pos));

    pos += shardSize_;
  }
  for(auto&& t : threads)
    t.join();
}

void AsyncGraphGroup::updateMovingAverage(Tensor paramsAvg,
                                          Tensor params,
                                          size_t batches) {
  using namespace functional;
  float decay
      = std::max(mvDecay_, 1.f - (float)(batches + 1) / (float)(batches + 10));
  Element(_1 = ((1.f - decay) * _1) + (decay * _2), paramsAvg, params);
}

void AsyncGraphGroup::init(Ptr<data::Batch> batch) {
  // initialize the parameters

  {
    ThreadPool pool(graphs_.size(), graphs_.size());
    for(size_t i = 0; i < graphs_.size(); ++i) {
      auto init = [&](size_t i) {
        builders_[i]->build(graphs_[i], batch);
        graphs_[i]->forward();
      };
      pool.enqueue(init, i);
    }
  }

  if(params_.size() == 0) {
    int totalSize = graphs_[0]->params()->vals()->size();
    shardSize_ = ceil(totalSize / (float)devices_.size());

    int pos = 0;
    // parameter sharding
    for(auto graph : graphs_) {
      int __size__ = std::min(shardSize_, totalSize);
      totalSize -= __size__;

      Tensor param;
      Ptr<TensorAllocator> allocator
          = New<TensorAllocator>(graph->getBackend());
      allocator->reserveExact(__size__ * sizeof(float));
      allocator->allocate(param, {1, __size__});
      paramsAlloc_.push_back(allocator);

      param->copyFrom(graphs_[0]->params()->vals()->subtensor(pos, __size__));
      params_.push_back(param);

      pos += __size__;
    }
  }
  if(grads_.size() == 0) {
    int totalSize = graphs_[0]->params()->vals()->size();

    for(auto graph : graphs_) {
      int __size__ = std::min(shardSize_, totalSize);
      totalSize -= __size__;
      Tensor grad_;
      Ptr<TensorAllocator> allocator_
          = New<TensorAllocator>(graph->getBackend());

      allocator_->reserveExact(__size__ * sizeof(float));
      allocator_->allocate(grad_, {1, __size__});
      gradsAlloc_.push_back(allocator_);
      grads_.push_back(grad_);
    }
  }
  if(movingAvg_) {
    if(paramsAvg_.size() == 0) {
      int totalSize = graphs_[0]->params()->vals()->size();

      int i = 0;
      for(auto graph : graphs_) {
        int __size__ = std::min(shardSize_, totalSize);
        totalSize -= __size__;
        Tensor paramAvg;
        Ptr<TensorAllocator> allocator
            = New<TensorAllocator>(graph->getBackend());

        allocator->reserveExact(__size__ * sizeof(float));
        allocator->allocate(paramAvg, {1, __size__});

        paramAvg->copyFrom(params_[i++]);

        paramsAllocAvg_.push_back(allocator);
        paramsAvg_.push_back(paramAvg);
      }
    }
  }
}

void AsyncGraphGroup::execute(Ptr<data::Batch> batch) {
  if(first_) {
    init(batch);
    first_ = false;
  }

  auto task = [this](Ptr<data::Batch> batch) {
    static size_t i = 0;
    thread_local Ptr<ExpressionGraph> graph;
    thread_local Ptr<models::ModelBase> builder;
    thread_local size_t t = 0;
    thread_local size_t num_seen_words = 0;
    thread_local size_t num_seen_trg = 0;
    thread_local size_t num_seen_sentences = 0;
    thread_local int t_id = 0;
    thread_local float cost = 0;

    thread_local Tensor accGradients;
    thread_local Ptr<TensorAllocator> accAlloc;

    if(!graph) {
      std::lock_guard<std::mutex> lock(sync_);
      t_id = i;
      graph = graphs_[i];
      builder = builders_[i++];
    }

    auto costNode = builder->build(graph, batch);

    if(t % optimizerDelay_ == 0) {
      fetchParams(graph->params()->vals(), params_, t_id);
    }

    graph->forward();
    cost += costNode->scalar();
    graph->backward();

    Tensor gradients;
    if(optimizerDelay_ > 1) {
      if(t == 0) {
        accAlloc = New<TensorAllocator>(graph->getBackend());
        accAlloc->reserveExact(graph->params()->grads()->memory()->size());
        accAlloc->allocate(accGradients, graph->params()->grads()->shape());
        accGradients->set(0);
      }

      using namespace functional;
      Element(_1 += _2, accGradients, graph->params()->grads());
      gradients = accGradients;

      // Keep track of how many words we've calculated the error from
      num_seen_words += batch->words();
      num_seen_trg += batch->wordsTrg();
      num_seen_sentences += batch->size();
    } else {
      gradients = graph->params()->grads();
      num_seen_trg = batch->wordsTrg();
    }

    t++;

    if(t % optimizerDelay_ == 0) {
      pushGradients(gradients, num_seen_trg, t_id);
      // Reset the counter of seen target words after gradient update
      num_seen_trg = 0;
      if(optimizerDelay_ > 1)
        gradients->set(0);
    }

    if(t % optimizerDelay_ == 0 && scheduler_) {
      std::unique_lock<std::mutex> lock(schedulerMutex_);

      // Wait until the thread that wants to do validation is finished.
      pool_->wait_for_one(lock);

      if (options_->get<std::string>("cost-type") != "ce-sum")
        cost /= optimizerDelay_;

      if (optimizerDelay_ > 1) {
        std::vector<size_t> fakeLength = {1, 1};
        auto fb = data::CorpusBatch::fakeBatch(fakeLength,
                                          num_seen_sentences,
                                          NULL);
        fb->front()->setWords(num_seen_words);
        scheduler_->update(cost, fb);

        num_seen_words = 0;
        num_seen_sentences = 0;
      } else {
        scheduler_->update(cost, batch);
      }

      cost = 0;

      if(scheduler_->saving() || scheduler_->validating()) {
        // Wait with validation or saving until all other threads are done with
        // update.
        // We want to reuse the graphs for validation, so they need to be in
        // a safe state.
        pool_->wait_for_others(lock);

        if(movingAvg_)
          for(auto g : graphs_)
            fetchParams(g->params()->vals(), paramsAvg_, t_id);

        if(scheduler_->saving())
          this->save(graph);

        if(scheduler_->validating())
          scheduler_->validate(graphs_);

        // Validation or saving is done, tell other threads to continue work.
        pool_->notify_others();
      }
    }
  };

  pool_->enqueue(task, batch);
}

void AsyncGraphGroup::finalize() {
  pool_->join_all(); // call before destructing thread pool
  pool_.reset(nullptr);
  finalized_ = true;
}

}
