#pragma once

#include <future>
#include <thread>

#include <boost/filesystem.hpp>
#include <boost/thread/locks.hpp>
#include <boost/thread/shared_mutex.hpp>

#include "3rd_party/threadpool.h"
#include "common/definitions.h"
#include "data/batch_generator.h"
#include "models/model_base.h"
#include "optimizers/optimizers.h"
#include "training/dropper.h"
#include "training/scheduler.h"
#include "training/sparse_tensor.h"
#include "training/training.h"
#include "training/validator.h"
#include "training/graph_group.h"

namespace marian {

template <class Builder>
class SyncGraphGroup : public GraphGroup {
public:
  typedef Builder builder_type;
  typedef typename Builder::dataset_type dataset_type;

  virtual void setScheduler(Ptr<Scheduler<dataset_type>> scheduler) {
    scheduler_ = scheduler;
    // optimizer has to be registered last to see a change of learning rate
    scheduler_->registerTrainingObserver(scheduler_);

    for(auto opt : shardOpt_)
      scheduler_->registerTrainingObserver(opt);
  }

private:
  std::vector<Ptr<models::ModelBase>> builders_;
  std::vector<Ptr<ExpressionGraph>> graphs_;
  std::vector<size_t> devices_;

  std::vector<Tensor> params_;
  std::vector<Tensor> grads_;
  std::vector<Tensor> tmpTensors_;
  std::vector<Ptr<TensorAllocator>> paramsAllocs_;

  std::vector<Ptr<OptimizerBase>> shardOpt_;

  int shardSize_;
  bool first_{true};

  Ptr<Scheduler<dataset_type>> scheduler_;

  void execute(Ptr<data::Batch> batch) {
    std::vector<Ptr<data::Batch>> batches = batch->split(devices_.size());

    if(first_) {

      for(size_t i = 0; i < graphs_.size(); ++i) {
        // takes care of thead_local stuff
        THREAD_GUARD(
          builders_[i]->build(graphs_[i], batches[0]);
          graphs_[i]->forward();
        );

        if(i > 0)
          graphs_[i]->params()->vals()->copyFrom(graphs_[0]->params()->vals());
      }

      if(params_.size() == 0) {
        int totalSize = graphs_[0]->params()->vals()->size();
        shardSize_ = ceil(totalSize / (float)devices_.size());

        int pos = 0;
        for(auto device : devices_) {
          int __size__ = min(shardSize_, totalSize);

          auto paramsAlloc = New<TensorAllocator>(device);
          paramsAllocs_.push_back(paramsAlloc);

          paramsAlloc->reserveExact(3 * __size__ * sizeof(float));

          Tensor param, grad, tmp;
          paramsAlloc->allocate(param, {__size__});
          paramsAlloc->allocate(grad, {__size__});
          paramsAlloc->allocate(tmp, {__size__});
          params_.push_back(param);
          grads_.push_back(grad);
          tmpTensors_.push_back(tmp);

          param->copyFrom(graphs_[0]->params()->vals()->subtensor(pos, __size__));
          pos += __size__;
          totalSize -= __size__;
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
        for(auto graph : graphs_) {
          if(batches[i]->size() > 0) {
            auto subGrad = graph->params()->grads()->subtensor(pos, size);
            tmpTensors_[idx]->copyFrom(subGrad);
            Element(_1 += _2, grads_[idx], tmpTensors_[idx]);
          }
          i++;
        }

        shardOpt_[idx]->update(params_[idx], grads_[idx]);

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

      if(scheduler_->saving())
        this->save();

      if(scheduler_->validating()) {
        scheduler_->validate(graphs_[0]);
      }
    }
  }

public:
  template <class... Args>
  SyncGraphGroup(Ptr<Config> options, Args... args)
      : GraphGroup(options),
        devices_{options_->get<std::vector<size_t>>("devices")}
  {
    for(auto device : devices_) {
      auto graph = New<ExpressionGraph>();
      graph->setDevice(device);
      graph->reserveWorkspaceMB(options_->get<size_t>("workspace"));
      graphs_.push_back(graph);
      shardOpt_.push_back(Optimizer(options_));
      builders_.push_back(models::from_config(options_));
    }
  }

  void update(Ptr<data::Batch> batch) { execute(batch); }

  void load() {
    if(!options_->get<bool>("no-reload")) {
      std::string init = options_->get<std::string>("model");
      if(boost::filesystem::exists(init)) {
        size_t i = 0;
        if(scheduler_)
          scheduler_->load(init);
        for(auto graph : graphs_)
          builders_[i++]->load(graph, init);
      }
    }
  }

  void save(bool final = false) { save(graphs_[0], final); }

  void save(Ptr<ExpressionGraph> graph, bool final = false) {
    int idx = 0;
    for(int i = 0; i < graphs_.size(); ++i) {
      if(graph == graphs_[i]) {
        idx = i;
        break;
      }
    }

    if(options_->get<bool>("overwrite")) {
      std::string name = options_->get<std::string>("model");

      builders_[idx]->save(graphs_[idx], name, true);
      if(scheduler_)
        scheduler_->save(name);
    } else {
      std::string name = options_->get<std::string>("model");

      if(!final) {
        std::string numberOfBatches
            = scheduler_ ? std::to_string(scheduler_->numberOfBatches()) :
                           "unknown";
        std::string nameOverwrite = name;
        nameOverwrite.replace(
            name.size() - 4, 4, ".iter" + numberOfBatches + ".npz");
        builders_[idx]->save(graphs_[idx], nameOverwrite);
      }

      builders_[idx]->save(graphs_[idx], name, true);
      if(scheduler_)
        scheduler_->save(name);
    }
  }

  Ptr<data::BatchStats> collectStats() {
    return builders_[0]->collectStats(graphs_[0], devices_.size());
  }
};

}
