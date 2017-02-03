#pragma once

#include <thread>

#include "common/definitions.h"

namespace marian {
  
class Reporter {
  private:
    Ptr<Config> options_;
    
    float costSum;
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
class SynchronousGraphGroup {
  private:
    Ptr<Config> options_;
    Ptr<OptimizerBase> opt_;
    Ptr<Builder> builder_;
    std::vector<Ptr<ExpressionGraph>> graphs_;
    Ptr<Reporter> reporter_;
  
    std::vector<Ptr<data::CorpusBatch>> batches_;
    
    void accumulateGradients(Ptr<ExpressionGraph> master,
                             std::vector<Ptr<ExpressionGraph>> graphs) {
      if(graphs_.size() < 2)
        return;
      
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
    }
  
    void distributeParameters(Ptr<ExpressionGraph> master,
                              std::vector<Ptr<ExpressionGraph>> graphs) {
      if(graphs_.size() < 2)
        return;
      
      Tensor params = master->params().vals();
      //std::vector<float> temp;
      //params->get(temp);
          
      for(auto graph : graphs) {
        if(graph != master) {
          graph->params().vals()->copyFrom(params);
          //graph->params().vals()->set(temp);
        }
      }
    }
  
    void execute() {
      float costSum = 0;
      std::vector<std::thread> tasks;
      
      auto task = [this, &costSum](Ptr<ExpressionGraph> graph,
                                   Ptr<data::CorpusBatch> batch,
                                   Ptr<Reporter> reporter) {
        this->builder_->build(graph, batch);
        graph->forward();
        float cost = graph->topNode()->scalar();
        graph->backprop();
        
        if(reporter)
          reporter->update(cost, batch);
      };
      
      for(int i = 0; i < batches_.size(); ++i)
        tasks.emplace_back(task, graphs_[i], batches_[i], reporter_);
      
      for(auto& thread : tasks)
        if(thread.joinable())
          thread.join();
          
      accumulateGradients(graphs_[0], graphs_);
      opt_->updateRule(graphs_[0]);
      distributeParameters(graphs_[0], graphs_);
      
      batches_.clear();
    }
  
  public:
    SynchronousGraphGroup(Ptr<Config> options)
     : options_(options) {
    
      builder_ = New<Builder>(options_);
      
      Ptr<ClipperBase> clipper = nullptr;
      float clipNorm = options_->get<double>("clip-norm");
      float lrate = options_->get<double>("lrate");
      if(clipNorm > 0)
        clipper = Clipper<Norm>(clipNorm);
      opt_ = Optimizer<Adam>(lrate, keywords::clip=clipper);
  
      for(auto device : options_->get<std::vector<size_t>>("device")) {
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
};
  
}