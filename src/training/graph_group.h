#pragma once

#include <future>
#include <thread>

#include <boost/filesystem.hpp>
#include <boost/thread/locks.hpp>
#include <boost/thread/shared_mutex.hpp>

#include "3rd_party/threadpool.h"
#include "common/definitions.h"
#include "data/batch_generator.h"
#include "optimizers/optimizers.h"
#include "training/dropper.h"
#include "training/scheduler.h"
#include "training/sparse_tensor.h"
#include "training/training.h"
#include "training/validator.h"

namespace marian {

class GraphGroup {
protected:
  Ptr<Config> options_;
  Ptr<OptimizerBase> opt_;

public:
  GraphGroup(Ptr<Config> options)
      : options_(options), opt_(Optimizer(options)) {}

  virtual ~GraphGroup() {}

  virtual void update(Ptr<data::Batch>) = 0;

  virtual void load() = 0;

  virtual void save(bool = false) = 0;

  virtual Ptr<data::BatchStats> collectStats() = 0;
};

template <class Builder>
class SingletonGraph : public GraphGroup {
public:
  typedef Builder builder_type;
  typedef typename Builder::dataset_type dataset_type;

  virtual void setScheduler(Ptr<Scheduler<dataset_type>> scheduler) {
    scheduler_ = scheduler;
    // optimizer has to be registered last to see a change of learning rate
    scheduler_->registerTrainingObserver(scheduler_);
    scheduler_->registerTrainingObserver(opt_);
  }

private:
  Ptr<Builder> builder_;
  Ptr<ExpressionGraph> graph_;

  Ptr<Scheduler<dataset_type>> scheduler_;

  Ptr<ExpressionGraph> mvAvgGraph_;
  bool mvAvg_{false};
  float mvDecay_{0.9999};

  void updateMovingAverage(Tensor mvAvgParams, Tensor params, size_t batches) {
    float decay = min(mvDecay_, (float)(batches + 1) / (float)(batches + 10));
    Element(_1 = (decay * _1) + ((1.f - decay) * _2), mvAvgParams, params);
  }

  void execute(Ptr<data::Batch> batch) {
    auto costNode = builder_->build(graph_, batch);

    graph_->forward();
    float cost = costNode->scalar();
    graph_->backward();

    opt_->update(graph_);

    if(mvAvg_) {
      if(!mvAvgGraph_) {
        mvAvgGraph_ = New<ExpressionGraph>();
        mvAvgGraph_->setDevice(graph_->getDevice());
        mvAvgGraph_->copyParams(graph_);
      } else {
        updateMovingAverage(mvAvgGraph_->params()->vals(),
                            graph_->params()->vals(),
                            scheduler_->numberOfBatches());
      }
    }

    if(scheduler_) {
      scheduler_->update(cost, batch);

      if(scheduler_->saving())
        this->save();

      if(scheduler_->validating()) {
        if(mvAvg_)
          scheduler_->validate(mvAvgGraph_);
        else
          scheduler_->validate(graph_);
      }

      /*if(mvAvg_) {
        size_t injectFreq = options_->get<size_t>("moving-inject-freq");
        if(injectFreq && scheduler_->numberOfBatches() % injectFreq == 0) {
          LOG(info)->info("{} : Injecting moving average into training parameters",
                          scheduler_->numberOfBatches());
          graph_->params()->vals()->copyFrom(mvAvgGraph_->params()->vals());
        }
      }*/
    }
  }

public:
  template <class... Args>
  SingletonGraph(Ptr<Config> options, Args... args)
      : GraphGroup(options),
        mvAvg_{options_->get<bool>("moving-average")},
        mvDecay_{(float)options_->get<double>("moving-decay")} {
    size_t device = options_->get<std::vector<size_t>>("devices")[0];

    graph_ = New<ExpressionGraph>();
    graph_->setDevice(device);
    graph_->reserveWorkspaceMB(options_->get<size_t>("workspace"));
    opt_ = Optimizer(options_);

    builder_ = New<Builder>(options_, args...);
  }

  void update(Ptr<data::Batch> batch) { execute(batch); }

  void load() {
    if(!options_->get<bool>("no-reload")) {
      std::string name = options_->get<std::string>("model");

      if(boost::filesystem::exists(name)) {
        if(scheduler_)
          scheduler_->load(name);
        builder_->load(graph_, name);
      }
    }
  }

  void save(bool final = false) {
    auto saveGraph = graph_;
    if(mvAvg_)
      saveGraph = mvAvgGraph_;

    save(saveGraph, final);
  }

  void save(Ptr<ExpressionGraph> graph, bool final = false) {
    if(options_->get<bool>("overwrite")) {
      std::string name = options_->get<std::string>("model");

      builder_->save(graph_, name, true);
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
        builder_->save(graph_, nameOverwrite);
      }

      builder_->save(graph_, name, true);
      if(scheduler_)
        scheduler_->save(name);
    }
  }

  Ptr<data::BatchStats> collectStats() {
    return builder_->collectStats(graph_);
  }
};

template <class Builder>
class AsyncGraphGroup : public GraphGroup {
public:
  typedef Builder builder_type;
  typedef typename Builder::dataset_type dataset_type;

  virtual void setScheduler(Ptr<Scheduler<dataset_type>> scheduler) {
    scheduler_ = scheduler;
    // optimizer has to be registered last to see a change of learning rate
    scheduler_->registerTrainingObserver(scheduler_);
    scheduler_->registerTrainingObserver(opt_);
  }

private:
  bool first_{true};

  std::vector<Ptr<Builder>> builders_;
  std::vector<Ptr<ExpressionGraph>> graphs_;
  std::vector<size_t> devices_;

  Ptr<Scheduler<dataset_type>> scheduler_;

  std::mutex sync_;
  std::vector<std::mutex> shardSync_;

  boost::shared_mutex schedulerMutex_;

  std::vector<SparseTensor> localSparseGrads_;
  std::vector<SparseTensor> sparseGrads_;
  std::vector<SparseTensor> tmpSparseDelta;
  std::vector<std::vector<SparseTensor>> localSparseDelta;

  // version number per-shard
  std::vector<int> globalVersionNumber;

  // each worker has the version number obtained from each shard
  std::vector<std::vector<int>> localVersionNumbers;

  std::vector<std::vector<GradientDrop>> fetchDropper;
  std::vector<Tensor> tmpTensor;

  std::vector<std::vector<Tensor>> params_;
  std::vector<Ptr<TensorAllocator>> paramsAlloc_;

  std::vector<Tensor> grads_;
  std::vector<Ptr<TensorAllocator>> gradsAlloc_;

  std::vector<Ptr<OptimizerBase>> shardOpt_;

  int shardSize_;
  int tau_{1};

  std::vector<Tensor> paramsAvg_;
  std::vector<Ptr<TensorAllocator>> paramsAllocAvg_;
  bool movingAvg_{false};
  float mvDecay_{0.9999};

  ThreadPool pool_;

  double drop_rate_{0};
  int history_size_{1};

  std::vector<Ptr<TensorAllocator>> allocators;

  Tensor newTensor(int size, int device) {
    Tensor t;
    Ptr<TensorAllocator> allocator_ = New<TensorAllocator>(device);
    allocator_->reserveExact(size * sizeof(float));
    allocator_->allocate(t, {1, size});
    allocators.push_back(allocator_);

    return t;
  }

  void fetchParams(Tensor oldParams, const std::vector<Tensor>& params) {
    // @TODO read guard on parameters
    int pos = 0;

    std::vector<std::thread> threads;
    for(int idx = 0; idx < devices_.size(); idx++) {
      threads.emplace_back(std::thread(
          [=](int idx, int pos) {
            // individual mutex per-shard
            std::lock_guard<std::mutex> guard(shardSync_[idx]);
            oldParams->subtensor(pos, params[idx]->size())
                ->copyFrom(params[idx]);
          },
          idx,
          pos));

      pos += shardSize_;
    }
    for(auto&& t : threads) {
      t.join();
    }
  }

  void pushGradients(Tensor newGrads) {
    // add instead of copy?
    std::vector<std::thread> threads;
    int pos = 0;
    for(int idx = 0; idx < devices_.size(); idx++) {
      threads.emplace_back(std::thread(
          [=](int idx, int pos) {
            // individual mutex per-shard
            std::lock_guard<std::mutex> guard(shardSync_[idx]);
            grads_[idx]->copyFrom(
                newGrads->subtensor(pos, grads_[idx]->size()));

            // apply and increment your version number, if history is enabled
            int latestVersion = 0;

            if(history_size_ > 1) {
              int pastVersion = globalVersionNumber[idx] % history_size_;
              latestVersion = ++globalVersionNumber[idx] % history_size_;
              params_[latestVersion][idx]->copyFrom(params_[pastVersion][idx]);
            }

            shardOpt_[idx]->update(params_[latestVersion][idx], grads_[idx]);

            if(movingAvg_)
              updateMovingAverage(paramsAvg_[idx], params_[latestVersion][idx],
                                  scheduler_->numberOfBatches());

            cudaStreamSynchronize(0);
          },
          idx,
          pos));

      pos += shardSize_;
    }
    for(auto&& t : threads)
      t.join();
  }

  void sparseFetchParams(Tensor oldParams, int worker_id) {
    if(graphs_.size() < 2)
      return;

    // @TODO read guard on parameters
    int p = 0;

    std::vector<std::thread> threads;
    for(int i = 0; i < devices_.size(); i++) {
      threads.emplace_back(std::thread(
          [=](int idx, int pos) {
            // individual mutex per-shard
            std::lock_guard<std::mutex> guard(shardSync_[idx]);
            // obtain the delta
            int latestVersion = globalVersionNumber[idx] % history_size_;
            int currVersion
                = localVersionNumbers[worker_id][idx] % history_size_;

            // check if the current version is too old
            if(globalVersionNumber[idx] - localVersionNumbers[worker_id][idx]
               >= history_size_)
              currVersion = (1 + globalVersionNumber[idx])
                            % history_size_;  // if so, pick the best you can do

            // if already latest
            if(globalVersionNumber[idx] == localVersionNumbers[worker_id][idx])
              return;

            // get delta : param latest version - current param (locally)
            Element(_1 = _2 - _3,
                    tmpTensor[idx],
                    params_[latestVersion][idx],
                    params_[currVersion][idx]);
            cudaStreamSynchronize(0);

            // get sparse delta
            fetchDropper[worker_id][idx]->dropGraph(
                tmpTensor[idx], tmpSparseDelta[idx], drop_rate_);
            cudaStreamSynchronize(0);

            // move sparse delta
            localSparseDelta[worker_id][idx]->copyFrom(tmpSparseDelta[idx]);
            cudaStreamSynchronize(0);

            localSparseDelta[worker_id][idx]->scatterAdd(
                oldParams->subtensor(pos, grads_[idx]->size()));
            cudaStreamSynchronize(0);

            localVersionNumbers[worker_id][idx] = globalVersionNumber[idx];

          },
          i,
          p));

      p += shardSize_;
    }
    for(auto&& t : threads) {
      t.join();
    }
  }

  void sparsePush(SparseTensor newGrads) {
    if(graphs_.size() < 2) {
      opt_->update(graphs_[0]);
    } else {
      // add instead of copy?
      std::vector<std::thread> threads;
      int pos = 0;
      for(int idx = 0; idx < devices_.size(); idx++) {
        threads.emplace_back(std::thread(
            [=](int idx, int pos) {
              // individual mutex per-shard
              std::lock_guard<std::mutex> guard(shardSync_[idx]);

              // split to shard
              SparseTensor subGrad
                  = newGrads->subtensor(pos, grads_[idx]->size(), idx);
              cudaStreamSynchronize(0);

              // sent
              sparseGrads_[idx]->copyFrom(subGrad);
              cudaStreamSynchronize(0);

              // convert back to dense, with index offset of -pos
              sparseGrads_[idx]->toDense(grads_[idx], -pos);
              cudaStreamSynchronize(0);

              // apply and increment your version number
              int pastVersion = globalVersionNumber[idx] % history_size_;
              int latestVersion = ++globalVersionNumber[idx] % history_size_;
              params_[latestVersion][idx]->copyFrom(params_[pastVersion][idx]);
              shardOpt_[idx]->update(params_[latestVersion][idx], grads_[idx]);

              if(movingAvg_)
                updateMovingAverage(paramsAvg_[idx],
                                    params_[latestVersion][idx],
                                    scheduler_->numberOfBatches());

              cudaStreamSynchronize(0);
            },
            idx,
            pos));

        pos += shardSize_;
      }
      for(auto&& t : threads)
        t.join();
    }
  }

  void updateMovingAverage(Tensor paramsAvg, Tensor params, size_t batches) {
    float decay = min(mvDecay_, (float)(batches + 1) / (float)(batches + 10));
    Element(_1 = (decay * _1) + ((1.f - decay) * _2), paramsAvg, params);
  }

  void execute(Ptr<data::Batch> batch) {
    if(first_) {
      // initialize the parameters
      for(size_t i = 0; i < graphs_.size(); ++i) {
        // takes care of thead_local stuff
        THREAD_GUARD(builders_[i]->build(graphs_[i], batch);
                     graphs_[i]->forward(););

        globalVersionNumber.push_back(0);
        std::vector<int> localVersion;
        for(int j = 0; j < graphs_.size(); j++)
          localVersion.push_back(0);

        localVersionNumbers.push_back(localVersion);
      }

      if(params_[0].size() == 0) {
        int totalSize = graphs_[0]->params()->vals()->size();
        shardSize_ = ceil(totalSize / devices_.size());

        int pos = 0;
        // parameter sharding
        for(auto device : devices_) {
          int __size__ = min(shardSize_, totalSize);
          totalSize -= __size__;

          for(int h_id = 0; h_id < history_size_; h_id++) {
            Tensor param;
            Ptr<TensorAllocator> allocator = New<TensorAllocator>(device);
            allocator->reserveExact(__size__ * sizeof(float));
            allocator->allocate(param, {1, __size__});
            paramsAlloc_.push_back(allocator);

            param->copyFrom(
                graphs_[0]->params()->vals()->subtensor(pos, __size__));
            params_[h_id].push_back(param);
          }

          if(drop_rate_)
            tmpTensor.push_back(newTensor(__size__, device));
          pos += __size__;
        }
      }
      if(grads_.size() == 0) {
        int totalSize = graphs_[0]->params()->vals()->size();

        for(auto device : devices_) {
          int __size__ = min(shardSize_, totalSize);
          totalSize -= __size__;
          Tensor grad_;
          Ptr<TensorAllocator> allocator_ = New<TensorAllocator>(device);

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
          for(auto device : devices_) {
            int __size__ = min(shardSize_, totalSize);
            totalSize -= __size__;
            Tensor paramAvg;
            Ptr<TensorAllocator> allocator = New<TensorAllocator>(device);

            allocator->reserveExact(__size__ * sizeof(float));
            allocator->allocate(paramAvg, {1, __size__});

            paramAvg->copyFrom(params_[0][i++]);

            paramsAllocAvg_.push_back(allocator);
            paramsAvg_.push_back(paramAvg);
          }
        }
      }

      if(drop_rate_ && first_) {
        int totalSize = graphs_[0]->params()->vals()->size();
        int sparseCap = totalSize * 1.2 * (1.0 - drop_rate_);
        for(auto device : devices_) {
          sparseGrads_.push_back(
              SparseTensor(new SparseTensorBase(sparseCap, device)));
          localSparseGrads_.push_back(
              SparseTensor(new SparseTensorBase(sparseCap, device)));
          tmpSparseDelta.push_back(SparseTensor(
              new SparseTensorBase(sparseCap / devices_.size(), device)));
          std::vector<SparseTensor> tmp;
          for(int i = 0; i < devices_.size(); i++)
            tmp.push_back(SparseTensor(
                new SparseTensorBase(sparseCap / devices_.size(), device)));
          localSparseDelta.push_back(tmp);
        }
      }

      first_ = false;
    }

    auto task = [this](Ptr<data::Batch> batch) {
      static size_t i = 0;
      thread_local Ptr<ExpressionGraph> graph;
      thread_local Ptr<Builder> builder;
      thread_local size_t t = 0;

      // gradient drop purpose
      thread_local GradientDrop dropper;

      thread_local size_t my_id = 0;

      if(!graph) {
        std::lock_guard<std::mutex> lock(sync_);
        my_id = i;
        graph = graphs_[i];
        builder = builders_[i++];
      }

      if(!dropper) {
        std::lock_guard<std::mutex> lock(sync_);
        dropper = GradientDrop(new GradientDropBase());
        std::vector<GradientDrop> tmp;
        for(int i = 0; i < devices_.size(); i++)
          tmp.push_back(GradientDrop(new GradientDropBase()));
        fetchDropper.push_back(tmp);
      }

      auto costNode = builder->build(graph, batch);

      if(drop_rate_ && t > 0)
        sparseFetchParams(graph->params()->vals(), my_id);
      else
        fetchParams(graph->params()->vals(),
                    params_[globalVersionNumber[my_id] % history_size_]);

      graph->forward();
      float cost = costNode->scalar();
      graph->backward();

      t++;

      cudaStreamSynchronize(0);
      if(drop_rate_) {
        dropper->dropGraph(
            graph->params()->grads(), localSparseGrads_[my_id], drop_rate_);
        sparsePush(localSparseGrads_[my_id]);
      } else
        pushGradients(graph->params()->grads());

      if(scheduler_) {
        boost::upgrade_lock<boost::shared_mutex> lock(schedulerMutex_);
        {
          boost::upgrade_to_unique_lock<boost::shared_mutex> uniqueLock(lock);
          scheduler_->update(cost, batch);
        }

        if(scheduler_->saving()) {
          boost::upgrade_to_unique_lock<boost::shared_mutex> uniqueLock(lock);
          if(movingAvg_)
            fetchParams(graph->params()->vals(), paramsAvg_);
          this->save(graph);
        }

        if(scheduler_->validating()) {
          boost::upgrade_to_unique_lock<boost::shared_mutex> uniqueLock(lock);
          if(movingAvg_)
            fetchParams(graph->params()->vals(), paramsAvg_);
          scheduler_->validate(graph);
        }

        /*if(movingAvg_) {
          size_t injectFreq = options_->get<size_t>("moving-inject-freq");
          if(injectFreq && scheduler_->numberOfBatches() % injectFreq == 0) {
            boost::upgrade_to_unique_lock<boost::shared_mutex> uniqueLock(lock);

            LOG(info)->info("{} : Injecting moving average into training parameters",
                            scheduler_->numberOfBatches());
            for(int idx = 0; idx < paramsAvg_.size(); idx++) {
              std::lock_guard<std::mutex> guard(shardSync_[idx]);
              params_[my_id][idx]->copyFrom(paramsAvg_[idx]);
            }
          }
        }*/
      }
    };

    pool_.enqueue(task, batch);
  }

public:
  template <class... Args>
  AsyncGraphGroup(Ptr<Config> options, Args... args)
      : GraphGroup(options),
        devices_{options_->get<std::vector<size_t>>("devices")},
        pool_{devices_.size(), devices_.size()},
        shardSync_{devices_.size()},
        movingAvg_{options_->get<bool>("moving-average")},
        mvDecay_{(float)options_->get<double>("moving-decay")},
        drop_rate_{options_->get<double>("drop-rate")} {
    if(drop_rate_ > 0.0) {
      history_size_ = devices_.size() * 1.5;
    }
    for(int i = 0; i < history_size_; i++)
      params_.push_back(std::vector<Tensor>());
    for(auto device : devices_) {
      auto graph = New<ExpressionGraph>();
      graph->setDevice(device);
      graph->reserveWorkspaceMB(options_->get<size_t>("workspace"));
      graphs_.push_back(graph);
      shardOpt_.push_back(Optimizer(options_));
      builders_.push_back(New<Builder>(options_, args...));
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
    return builders_[0]->collectStats(graphs_[0]);
  }
};
}
