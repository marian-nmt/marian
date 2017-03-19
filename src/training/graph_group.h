#pragma once

#include <thread>
#include <future>
#include <boost/filesystem.hpp>

#include "common/definitions.h"
#include "3rd_party/threadpool.h"
#include "optimizers/optimizers.h"
#include "training/training.h"
#include "training/validator.h"

namespace marian {

class GraphGroup {
  protected:
    Ptr<Config> options_;
    Ptr<Reporter> reporter_;
    Ptr<OptimizerBase> opt_;

    std::vector<Ptr<ExpressionGraph>> graphs_;

  public:
    GraphGroup(Ptr<Config> options)
    : options_(options), opt_(Optimizer(options)) { }

    virtual void update(Ptr<data::CorpusBatch>) = 0;

    virtual void setReporter(Ptr<Reporter> reporter) {
      reporter_ = reporter;
    }

    virtual void load() = 0;

    virtual void save() = 0;
};


template <class Builder>
class AsyncGraphGroup : public GraphGroup {
  private:
    std::vector<Ptr<Builder>> builders_;

    std::vector<size_t> devices_;

    std::vector<Ptr<ExpressionGraph>> graphs_;

    std::mutex sync_;
    std::vector<std::mutex> shardSync_;

    std::vector<Tensor> params_;
    std::vector<Ptr<TensorAllocator> > paramsAlloc_;

    std::vector<Tensor> grads_;
    std::vector<Ptr<TensorAllocator>> gradsAlloc_;

    std::vector<Ptr<OptimizerBase>> shardOpt_;

    int shardSize_;

    ThreadPool pool_;

    void fetchParams(Tensor oldParams) {
      if(graphs_.size() < 2)
        return;

      // @TODO read guard on parameters
      int pos = 0;

      std::vector<std::thread> threads;
      for (int idx = 0; idx < devices_.size(); idx++) {
        threads.emplace_back( std::thread( [=](int idx, int pos) {
          //individual mutex per-shard
          std::lock_guard<std::mutex> guard( shardSync_[idx] );
          oldParams->subtensor(pos , params_[idx]->size())->copyFrom(params_[idx]);
        }, idx, pos) );

        pos += shardSize_;
      }
      for (auto &&t : threads) {
        t.join();
      }
    }

    void pushGradients(Tensor newGrads) {
      if(graphs_.size() < 2) {
        opt_->update(graphs_[0]);
      }
      else {
        // add instead of copy?
        std::vector<std::thread> threads;
        int pos = 0;
        for (int idx = 0; idx < devices_.size(); idx++) {
          threads.emplace_back( std::thread([=](int idx, int pos) {
            //individual mutex per-shard
            std::lock_guard<std::mutex> guard( shardSync_[idx] );
            grads_[idx]->copyFrom( newGrads->subtensor(pos , grads_[idx]->size() ) );
            shardOpt_[idx]->update(params_[idx], grads_[idx]);

            cudaStreamSynchronize(0);
          } , idx, pos) );

          pos += shardSize_;
        }
        for(auto&& t : threads)
          t.join();
      }
    }

    void execute(Ptr<data::CorpusBatch> batch) {
      static bool first = true;
      if(first && graphs_.size() > 1) {
        // initialize the parameters
        for(size_t i = 0; i < graphs_.size(); ++i) {
          builders_[i]->build(graphs_[i], batch);
          graphs_[i]->forward();
        }

        if(params_.size() == 0) {
          int totalSize = graphs_[0]->params().vals()->size();
          shardSize_ = ceil(totalSize / devices_.size());

          int pos = 0;
          //parameter sharding
          for (auto device : devices_){
            int __size__ = min(shardSize_, totalSize);
            totalSize -= __size__;
            Tensor param_;
            Ptr<TensorAllocator> allocator_ = New<TensorAllocator>(device);

            allocator_->reserveExact(__size__);
            allocator_->allocate(param_, {1, __size__});
            paramsAlloc_.push_back(allocator_);
            param_->copyFrom( graphs_[0]->params().vals()->subtensor( pos , __size__ ) );
            params_.push_back(param_);
            pos += __size__;

          }
        }
        if(grads_.size() == 0) {
          int totalSize = graphs_[0]->params().vals()->size();

          for (auto device : devices_){
            int __size__ = min(shardSize_, totalSize);
            totalSize -= __size__;
            Tensor grad_;
            Ptr<TensorAllocator> allocator_ = New<TensorAllocator>(device);

            allocator_->reserveExact(__size__);
            allocator_->allocate(grad_, {1, __size__});
            gradsAlloc_.push_back(allocator_);
            grads_.push_back(grad_);

          }
        }

        first = false;
      }

      auto task = [this](Ptr<data::CorpusBatch> batch) {
        static size_t i = 0;
        thread_local Ptr<ExpressionGraph> graph;
        thread_local Ptr<Builder> builder;
        thread_local size_t t = 0;

        if(!graph) {
          std::lock_guard<std::mutex> lock(sync_);
          graph = graphs_[i];
          builder = builders_[i++];
        }

        builder->build(graph, batch);
        fetchParams(graph->params().vals());

        graph->forward();
        float cost = graph->topNode()->scalar();
        graph->backward();

        cudaStreamSynchronize(0);
        pushGradients(graph->params().grads());

        if(reporter_) {
          std::lock_guard<std::mutex> guard(sync_);
          reporter_->update(cost, batch);
          if(reporter_->batches % options_->get<size_t>("save-freq") == 0)
            this->save();
          size_t prevStalled = reporter_->stalled();
          reporter_->validate(graph);
          if(prevStalled < reporter_->stalled())
            for(auto opt : shardOpt_)
              opt->updateSchedule();
        }

        t++;
      };

      pool_.enqueue(task, batch);
    }

  public:
    typedef Builder builder_type;

    AsyncGraphGroup(Ptr<Config> options)
     : GraphGroup(options),
       devices_{options_->get<std::vector<size_t>>("device")},
       pool_{devices_.size(), devices_.size()},
       shardSync_{devices_.size()} {

      for(auto device : devices_) {
        auto graph = New<ExpressionGraph>();
        graph->setDevice(device);
        graph->reserveWorkspaceMB(options_->get<size_t>("workspace"));
        graphs_.push_back(graph);
        shardOpt_.push_back(Optimizer(options_));
        builders_.push_back(New<Builder>(options_));
      }
    }

    void update(Ptr<data::CorpusBatch> batch) {
      execute(batch);
    }

    void load() {
      if(!options_->get<bool>("no-reload")) {
        std::string init = options_->get<std::string>("model");
        if(boost::filesystem::exists(init)) {
          size_t i = 0;
          reporter_->load(init);
          for(auto graph : graphs_)
            builders_[i++]->load(graph, init);
        }
      }
    }

    void save() {
      if(options_->get<bool>("overwrite")) {
        std::string name = options_->get<std::string>("model");
        builders_[0]->save(graphs_[0], name);
        reporter_->save(name);
      }
      else {
        std::string name = options_->get<std::string>("model");
        std::string nameOverwrite = name;
        nameOverwrite.replace(name.size() - 4, 4,
          ".iter" + std::to_string(reporter_->batches) + ".npz");
        builders_[0]->save(graphs_[0], nameOverwrite);
        reporter_->save(nameOverwrite);
        builders_[0]->save(graphs_[0], name);
        reporter_->save(name);
      }
    }
};


template <class Builder>
class SyncGraphGroup : public GraphGroup {
  private:
    Ptr<Builder> builder_;
    std::vector<Ptr<data::CorpusBatch>> batches_;

    bool first_{true};

    void accumulateGradients(Ptr<ExpressionGraph> master,
                             std::vector<Ptr<ExpressionGraph>> graphs) {
      if(graphs_.size() < 2) {
        return;
      }

      Tensor grads = master->params().grads();
      Tensor tempGrads;
      master->tensor(tempGrads, grads->shape());

      for(auto graph : graphs) {
        if(graph != master) {
          Tensor remoteGrads = graph->params().grads();
          tempGrads->copyFrom(remoteGrads);
          Element(_1 += _2, grads, tempGrads);
        }
      }

      float denom = graphs_.size();
      Element(_1 /= denom, grads);
    }

    void distributeParameters(Ptr<ExpressionGraph> master,
                              std::vector<Ptr<ExpressionGraph>> graphs) {
      if(graphs_.size() < 2)
        return;

      Tensor params = master->params().vals();
      for(auto graph : graphs) {
        if(graph != master) {
          graph->params().vals()->copyFrom(params);
        }
      }
    }

    void execute() {
      if(first_) {
        for(auto graph : graphs_) {
          builder_->build(graph, batches_[0]);
          graph->forward();
        }
        distributeParameters(graphs_[0], graphs_);
        first_ = false;
      }

      auto task = [this](int i,
                         Ptr<data::CorpusBatch> batch) {
        thread_local int j = -1;
        if(j == -1)
          j = i;
        auto localGraph = this->graphs_[j];

        builder_->build(localGraph, batch);
        localGraph->forward();
        float cost = localGraph->topNode()->scalar();
        localGraph->backward();

        if(reporter_) {
          reporter_->update(cost, batch);
          if(reporter_->batches % options_->get<size_t>("save-freq") == 0)
            this->save();
        }
      };

      {
        size_t workers = graphs_.size();
        ThreadPool pool(workers, workers);

        for(int i = 0; i < batches_.size(); ++i)
          pool.enqueue(task, i % (int)workers, batches_[i]);
      }
      accumulateGradients(graphs_[0], graphs_);
      opt_->update(graphs_[0]);
      distributeParameters(graphs_[0], graphs_);

      batches_.clear();
    }

    void load() {
      if(options_->has("init")) {
        std::string init = options_->get<std::string>("init");
        for(auto graph : graphs_)
        builder_->load(graph, init);
      }
    }

  public:
    typedef Builder builder_type;

    SyncGraphGroup(Ptr<Config> options)
     : GraphGroup(options),
       builder_{New<Builder>(options_)} {

      auto devices = options_->get<std::vector<size_t>>("device");
      size_t workers = devices.size();

      for(auto device : devices) {
        graphs_.emplace_back(New<ExpressionGraph>());
        graphs_.back()->setDevice(device);
        graphs_.back()->reserveWorkspaceMB(options_->get<size_t>("workspace"));
      }

      load();
    }

    ~SyncGraphGroup() {
      execute();
    }

    void update(Ptr<data::CorpusBatch> batch) {
      batches_.push_back(batch);
      if(batches_.size() == graphs_.size())
        execute();
    }

    void save() {
      if(options_->get<bool>("overwrite")) {
        std::string name = options_->get<std::string>("model") + ".npz";
        builder_->save(graphs_[0], name);
      }
      else {
        std::string name = options_->get<std::string>("model")
          + "." + std::to_string(reporter_->batches) + ".npz";
        builder_->save(graphs_[0], name);
      }
    }

};

}
