#pragma once

#include <thread>

#include "common/definitions.h"
#include "3rd_party/threadpool.h"

namespace marian {
  
class Reporter {
  public:
    Ptr<Config> options_;
    
    float costSum{0};
    size_t epochs{1};
    
    size_t samples{0};
    size_t wordsDisp{0};
    size_t batches{0};
    
    boost::timer::cpu_timer timer;
    
  public:  
    Reporter(Ptr<Config> options) : options_(options) {}
    
    void update(float cost, Ptr<data::CorpusBatch> batch) {
      static std::mutex sMutex;
      std::lock_guard<std::mutex> guard(sMutex);

      costSum += cost;
      samples += batch->size();
      wordsDisp += batch->words();
      batches++;
      //if(options.get<size_t>("after-batches")
      //   && batches >= options.get<size_t>("after-batches"))
      //  break;
  
      if(batches % options_->get<size_t>("disp-freq") == 0) {
        std::stringstream ss;
        ss << "Ep. " << epochs
           << " : Up. " << batches
           << " : Sen. " << samples
           << " : Cost " << std::fixed << std::setprecision(2)
                         << costSum / options_->get<size_t>("disp-freq")
           << " : Time " << timer.format(2, "%ws");
  
        float seconds = std::stof(timer.format(5, "%w"));
        float wps = wordsDisp /   (float)seconds;
  
        ss << " : " << std::fixed << std::setprecision(2)
           << wps << " words/s";
  
        LOG(info) << ss.str();
  
        timer.start();
        costSum = 0;
        wordsDisp = 0;
      }
    }
};

template <class Builder>
class AsynchronousGraphGroup {
  private:
    Ptr<Config> options_;
    Ptr<OptimizerBase> opt_;
    Ptr<Builder> builder_;
    
    std::vector<Ptr<ExpressionGraph>> graphs_;
    
    Ptr<Reporter> reporter_;
    
    bool first_{true};
    
    void execute(Ptr<data::CorpusBatch> batch) {
      if(first_) {
        for(auto graph : graphs_) {
          builder_->build(graph, batches_[0]);
          graph->forward();
          // params_.copyFrom(graph->params.vals());
        }
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
  
  public:
    AsynchronousGraphGroup(Ptr<Config> options)
     : options_(options) {
      auto devices = options_->get<std::vector<size_t>>("device");
      size_t workers = devices.size();
      
      builder_ = New<Builder>(options_);
      
      Ptr<ClipperBase> clipper = nullptr;
      float clipNorm = options_->get<double>("clip-norm");
      float lrate = options_->get<double>("lrate");
      if(clipNorm > 0)
        clipper = Clipper<Norm>(clipNorm);
      
      opt_ = Optimizer<Adagrad>(lrate,
                                keywords::clip=clipper);
  
      for(auto device : devices) {
        graphs_.emplace_back(New<ExpressionGraph>());
        graphs_.back()->setDevice(device);
        graphs_.back()->reserveWorkspaceMB(options_->get<size_t>("workspace"));
      }
    }
    
    void setReporter(Ptr<Reporter> reporter) {
      reporter_ = reporter;
    }
    
    void update(Ptr<data::CorpusBatch> batch) {
      execute(batch);
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
  
template <class Builder>
class SynchronousGraphGroup {
  private:
    Ptr<Config> options_;
    Ptr<OptimizerBase> opt_;
    Ptr<Builder> builder_;
    std::vector<Ptr<ExpressionGraph>> graphs_;
    Ptr<Reporter> reporter_;
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
  
  public:
    SynchronousGraphGroup(Ptr<Config> options)
     : options_(options) {
      auto devices = options_->get<std::vector<size_t>>("device");
      size_t workers = devices.size();
      
      builder_ = New<Builder>(options_);
      
      Ptr<ClipperBase> clipper = nullptr;
      float clipNorm = options_->get<double>("clip-norm");
      float lrate = options_->get<double>("lrate");
      if(clipNorm > 0)
        clipper = Clipper<Norm>(clipNorm);
      
      opt_ = Optimizer<Adagrad>(lrate,
                                keywords::clip=clipper);
  
      for(auto device : devices) {
        graphs_.emplace_back(New<ExpressionGraph>());
        graphs_.back()->setDevice(device);
        graphs_.back()->reserveWorkspaceMB(options_->get<size_t>("workspace"));
      }
    }
    
    void setReporter(Ptr<Reporter> reporter) {
      reporter_ = reporter;
    }
    
    ~SynchronousGraphGroup() {
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