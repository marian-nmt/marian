#pragma once

#include <thread>
#include <future>
#include <boost/filesystem.hpp>
#include <boost/thread/locks.hpp>
#include <boost/thread/shared_mutex.hpp>

#include "common/definitions.h"
#include "3rd_party/threadpool.h"
#include "optimizers/optimizers.h"
#include "training/training.h"
#include "training/validator.h"
#include "data/batch_generator.h"

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
    
    virtual ~GraphGroup() {}

    virtual void update(Ptr<data::CorpusBatch>) = 0;

    virtual void setReporter(Ptr<Reporter> reporter) {
      reporter_ = reporter;
    }

    virtual void load() = 0;

    virtual void save(bool=false) = 0;
    
    virtual Ptr<data::BatchStats> collectStats() = 0;
};

template <class Builder>
class Singleton : public GraphGroup {
  private:
    Ptr<Builder> builder_;
    Ptr<ExpressionGraph> graph_;
    
    Ptr<ExpressionGraph> mvAvgGraph_;
    bool mvAvg_{false};
    float mvDecay_{0.999};
    
    void updateMovingAverage(Tensor mvAvgParams, Tensor params) {
      Element(_1 = (mvDecay_ * _1) + ((1.f - mvDecay_) * _2),
              mvAvgParams, params);
    }
    
    void execute(Ptr<data::CorpusBatch> batch) {          
      builder_->build(graph_, batch);
      
      graph_->forward();
      float cost = graph_->topNode()->scalar();
      graph_->backward();

      opt_->update(graph_);
      
      if(mvAvg_) {
        if(!mvAvgGraph_) {
          mvAvgGraph_ = New<ExpressionGraph>();
          mvAvgGraph_->setDevice(graph_->getDevice());
          mvAvgGraph_->reuseWorkspace(graph_);
          
          builder_->build(mvAvgGraph_, batch);
          mvAvgGraph_->forward();
          
          mvAvgGraph_->params()->vals()->copyFrom(graph_->params()->vals());
        }
        else {
          updateMovingAverage(mvAvgGraph_->params()->vals(),
                              graph_->params()->vals());
        }
      }
      
      if(reporter_) {
        reporter_->update(cost, batch);
        
        if(reporter_->saving())
          this->save();
         
        if(reporter_->validating())
          if(mvAvg_)
            reporter_->validate(mvAvgGraph_);
          else
            reporter_->validate(graph_);
      }
    }

  public:
    typedef Builder builder_type;

    Singleton(Ptr<Config> options)
     : GraphGroup(options),
       mvAvg_{options_->get<bool>("moving-average")},
       mvDecay_{(float)options_->get<double>("moving-decay")} {

      size_t device = options_->get<std::vector<size_t>>("devices")[0];
       
      graph_ = New<ExpressionGraph>();
      graph_->setDevice(device);
      graph_->reserveWorkspaceMB(options_->get<size_t>("workspace"));
      opt_ = Optimizer(options_);
      builder_ = New<Builder>(options_); 
    }

    void update(Ptr<data::CorpusBatch> batch) {
      execute(batch);
    }

    void load() {
      if(!options_->get<bool>("no-reload")) {
        std::string init = options_->get<std::string>("model");
        if(boost::filesystem::exists(init)) {
          reporter_->load(init);
          builder_->load(graph_, init);
        }
      }
    }
    
    void save(bool final=false) {
      auto saveGraph = graph_;
      if(mvAvg_)
        saveGraph = mvAvgGraph_;
        
      save(saveGraph, final);
    }
    
    void save(Ptr<ExpressionGraph> graph, bool final=false) {
      if(options_->get<bool>("overwrite")) {
        std::string name = options_->get<std::string>("model");
          
        builder_->save(graph_, name, true);
        reporter_->save(name);
      }
      else {
        std::string name = options_->get<std::string>("model");

        if(!final) {
          std::string nameOverwrite = name;
          nameOverwrite.replace(name.size() - 4, 4,
            ".iter" + std::to_string(reporter_->batches) + ".npz");
          builder_->save(graph_, nameOverwrite);
        }

        builder_->save(graph_, name, true);
        reporter_->save(name);
      }
    }
    
    Ptr<data::BatchStats> collectStats() {
      return builder_->collectStats(graph_);
    }
};


template <class Builder>
class AsyncGraphGroup : public GraphGroup {
  private:
    bool first_{true};
    
    std::vector<Ptr<Builder>> builders_;

    std::vector<size_t> devices_;

    std::vector<Ptr<ExpressionGraph>> graphs_;
    
    std::mutex sync_;
    std::vector<std::mutex> shardSync_;
    
    boost::shared_mutex reporterMutex_;

    std::vector<Tensor> params_;
    std::vector<Ptr<TensorAllocator>> paramsAlloc_;
  
    std::vector<Tensor> grads_;
    std::vector<Ptr<TensorAllocator>> gradsAlloc_;

    std::vector<Ptr<OptimizerBase>> shardOpt_;

    int shardSize_;
    int tau_{1};
  
    std::vector<Tensor> paramsAvg_;
    std::vector<Ptr<TensorAllocator>> paramsAllocAvg_;
    bool movingAvg_{false};
    float mvDecay_{0.999};

    ThreadPool pool_;

    void fetchParams(Tensor oldParams, const std::vector<Tensor>& params) {
      // @TODO read guard on parameters
      int pos = 0;

      std::vector<std::thread> threads;
      for (int idx = 0; idx < devices_.size(); idx++) {
        threads.emplace_back( std::thread( [=](int idx, int pos) {
          //individual mutex per-shard
          std::lock_guard<std::mutex> guard( shardSync_[idx] );
          oldParams->subtensor(pos , params[idx]->size())->copyFrom(params[idx]);
        }, idx, pos) );

        pos += shardSize_;
      }
      for (auto &&t : threads) {
        t.join();
      }
    }

    void pushGradients(Tensor newGrads) {
      // add instead of copy?
      std::vector<std::thread> threads;
      int pos = 0;
      for (int idx = 0; idx < devices_.size(); idx++) {
        threads.emplace_back( std::thread([=](int idx, int pos) {
          //individual mutex per-shard
          std::lock_guard<std::mutex> guard(shardSync_[idx]);
          grads_[idx]->copyFrom(newGrads->subtensor(pos, grads_[idx]->size()));
          shardOpt_[idx]->update(params_[idx], grads_[idx]);

          if(movingAvg_)
            updateMovingAverage(paramsAvg_[idx], params_[idx]);
          
          cudaStreamSynchronize(0);
        }, idx, pos));

        pos += shardSize_;
      }
      for(auto&& t : threads)
        t.join();    
    }

    void updateMovingAverage(Tensor paramsAvg, Tensor params) {
        Element(_1 = (mvDecay_ * _1) + ((1.f - mvDecay_) * _2),
                paramsAvg, params);
    }
    
    void execute(Ptr<data::CorpusBatch> batch) {      
      if(first_) {
        // initialize the parameters
        for(size_t i = 0; i < graphs_.size(); ++i) {
          builders_[i]->build(graphs_[i], batch);
          graphs_[i]->forward();
        }
        
        if(params_.size() == 0) {
          int totalSize = graphs_[0]->params()->vals()->size();
          shardSize_ = ceil(totalSize / devices_.size());

          int pos = 0;
          //parameter sharding
          for (auto device : devices_){
            int __size__ = min(shardSize_, totalSize);
            totalSize -= __size__;
            Tensor param;
            Ptr<TensorAllocator> allocator = New<TensorAllocator>(device);

            allocator->reserveExact(__size__);
            allocator->allocate(param, {1, __size__});
            paramsAlloc_.push_back(allocator);
            param->copyFrom(graphs_[0]->params()->vals()->subtensor(pos, __size__ ));
            params_.push_back(param);
            pos += __size__;
          }
        }
        if(grads_.size() == 0) {
          int totalSize = graphs_[0]->params()->vals()->size();

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
        if(movingAvg_) {
          if(paramsAvg_.size() == 0) {
            int totalSize = graphs_[0]->params()->vals()->size();
            
            int i = 0;
            for(auto device : devices_){
              int __size__ = min(shardSize_, totalSize);
              totalSize -= __size__;
              Tensor paramAvg;
              Ptr<TensorAllocator> allocator = New<TensorAllocator>(device);
  
              allocator->reserveExact(__size__);
              allocator->allocate(paramAvg, {1, __size__});
              
              paramAvg->copyFrom(params_[i++]);
              
              paramsAllocAvg_.push_back(allocator);
              paramsAvg_.push_back(paramAvg);
            }
          }
        }
        
        first_ = false;
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
        
        fetchParams(graph->params()->vals(), params_);
        
        graph->forward();
        float cost = graph->topNode()->scalar();
        graph->backward();

        t++;
        
        cudaStreamSynchronize(0);
        pushGradients(graph->params()->grads());

        if(reporter_) {
          boost::upgrade_lock<boost::shared_mutex> lock(reporterMutex_);
          {
            boost::upgrade_to_unique_lock<boost::shared_mutex> uniqueLock(lock);
            reporter_->update(cost, batch);
          }
          
          if(reporter_->saving()) {
            boost::upgrade_to_unique_lock<boost::shared_mutex> uniqueLock(lock);
            if(movingAvg_)
              fetchParams(graph->params()->vals(), paramsAvg_);
            this->save(graph);
          }
           
          if(reporter_->validating()) {
            boost::upgrade_to_unique_lock<boost::shared_mutex> uniqueLock(lock);
            if(movingAvg_)
              fetchParams(graph->params()->vals(), paramsAvg_);
            reporter_->validate(graph);
          }
        }
      };

      pool_.enqueue(task, batch);
    }

  public:
    typedef Builder builder_type;

    AsyncGraphGroup(Ptr<Config> options)
     : GraphGroup(options),
       devices_{options_->get<std::vector<size_t>>("devices")},
       pool_{devices_.size(), devices_.size()},
       shardSync_{devices_.size()},
       movingAvg_{options_->get<bool>("moving-average")},
       mvDecay_{(float)options_->get<double>("moving-decay")} {

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
    
    void save(bool final=false) {      
      save(graphs_[0], final);
    }
    
    void save(Ptr<ExpressionGraph> graph, bool final=false) {
      
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
        reporter_->save(name);
      }
      else {
        std::string name = options_->get<std::string>("model");

        if(!final) {
          std::string nameOverwrite = name;
          nameOverwrite.replace(name.size() - 4, 4,
            ".iter" + std::to_string(reporter_->batches) + ".npz");
          builders_[idx]->save(graphs_[idx], nameOverwrite);
        }

        builders_[idx]->save(graphs_[idx], name, true);
        reporter_->save(name);
      }
    }
    
    Ptr<data::BatchStats> collectStats() {
      return builders_[0]->collectStats(graphs_[0]);
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

      Tensor grads = master->params()->grads();
      Tensor tempGrads;
      master->tensor(tempGrads, grads->shape());

      for(auto graph : graphs) {
        if(graph != master) {
          Tensor remoteGrads = graph->params()->grads();
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

      Tensor params = master->params()->vals();
      for(auto graph : graphs) {
        if(graph != master) {
          graph->params()->vals()->copyFrom(params);
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

      auto devices = options_->get<std::vector<size_t>>("devices");
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

    void save(bool final=false) {
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
    
    Ptr<data::BatchStats> collectStats() {
      return builder_->collectStats(graphs_[0]);
    }
};

}
