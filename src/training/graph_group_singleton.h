#pragma once

#include <future>

#include <boost/filesystem.hpp>

#include "training/graph_group.h"

namespace marian {

/**
 * Single gpu training
 */
class SingletonGraph : public GraphGroup {
public:
  virtual void setScheduler(Ptr<Scheduler> scheduler);

private:
  Ptr<models::ModelBase> builder_;
  Ptr<ExpressionGraph> graph_;

  Ptr<ExpressionGraph> mvAvgGraph_;
  bool mvAvg_{false};
  float mvDecay_{1e-4};

  void updateMovingAverage(Tensor mvAvgParams, Tensor params, size_t batches);

  void execute(Ptr<data::Batch> batch);

public:
  SingletonGraph(Ptr<Config> config)
      : GraphGroup(config),
        mvAvg_{options_->get<float>("exponential-smoothing") > 0},
        mvDecay_{options_->get<float>("exponential-smoothing")} {
    auto deviceId = options_->getDevices()[0];
    graph_ = New<ExpressionGraph>();
    graph_->setDevice(deviceId);
    graph_->getBackend()->setClip(options_->get<float>("clip-gemm"));
    graph_->reserveWorkspaceMB(options_->get<size_t>("workspace"));
    opt_ = Optimizer(options_);
    builder_ = models::from_config(options_, models::usage::training);
  }

  void update(Ptr<data::Batch> batch) {
    ABORT_IF(finalized_, "Training has already finished.");
    execute(batch);
  }

  void load() {
    if(!options_->get<bool>("no-reload")) {
      std::string name = options_->get<std::string>("model");

      if(boost::filesystem::exists(name)) {
        if(scheduler_)
          scheduler_->load(name);
        builder_->load(graph_, name);

        opt_->load(name + ".optimizer.npz", {opt_}, {graph_->getBackend()});
      } else if(options_->has("pretrained-model")) {
        std::string init = options_->get<std::string>("pretrained-model");
        LOG(info,
            "Initialize model weights with the pre-trained model {}",
            init);
        builder_->load(graph_, init, false);
      }
    }
  }

  void save(bool final = false) {
    auto saveGraph = graph_;
    if(mvAvg_)
      saveGraph = mvAvgGraph_;

    if(final && scheduler_)
      scheduler_->validate({saveGraph}, true);

    save(saveGraph, final);
  }

  void save(Ptr<ExpressionGraph> graph, bool final = false) {
    std::string name = options_->get<std::string>("model");

    if(options_->get<bool>("overwrite")) {
      builder_->save(graph, name, true);
      if(scheduler_)
        scheduler_->save(name);
    } else {
      if(!final) {
        std::string numberOfBatches
            = scheduler_ ? std::to_string(scheduler_->numberOfBatches())
                         : "unknown";
        std::string nameOverwrite = name;
        nameOverwrite.replace(
            name.size() - 4, 4, ".iter" + numberOfBatches + ".npz");
        builder_->save(graph, nameOverwrite);
      }

      builder_->save(graph, name, true);
      if(scheduler_)
        scheduler_->save(name);
    }

    size_t totalSize = graph_->params()->vals()->size();
    opt_->save(name + ".optimizer.npz", {opt_}, totalSize);
  }

  Ptr<data::BatchStats> collectStats() {
    return GraphGroup::collectStats(graph_, builder_);
  }

  virtual void finalize() {
    finalized_ = true;
  }

};
}
