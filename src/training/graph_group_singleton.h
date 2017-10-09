#pragma once

#include <future>

#include <boost/filesystem.hpp>

#include "training/dropper.h"
#include "training/graph_group.h"
#include "training/sparse_tensor.h"
#include "training/validator.h"

namespace marian {

class SingletonGraph : public GraphGroup {
public:
  virtual void setScheduler(Ptr<Scheduler> scheduler) {
    scheduler_ = scheduler;
    // optimizer has to be registered last to see changes of learning rate
    scheduler_->registerTrainingObserver(scheduler_);
    scheduler_->registerTrainingObserver(opt_);
  }

private:
  Ptr<models::ModelBase> builder_;
  Ptr<ExpressionGraph> graph_;

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

    // Get batch stats
    size_t batch_words = batch->words();
    //@TODO use this to gather statistics about the usual number of words per batch
    //std::cout << "Batch size: " << batch->size() << " batch_words " << batch_words << std::endl;

    if (scaleLearningRate_) {
      opt_->update(graph_, batch_words/avgBatchWords_);
    } else {
      opt_->update(graph_);
    }

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

    builder_ = models::from_config(options_);
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

}
