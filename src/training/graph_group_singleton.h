#pragma once

#include <boost/filesystem.hpp>
#include <future>

#include "training/exponential_smoothing.h"
#include "training/graph_group.h"

namespace marian {

/**
 * Single GPU training
 */
class SingletonGraph : public GraphGroup, public ExponentialSmoothing {
public:
  virtual void setScheduler(Ptr<Scheduler> scheduler) override;

private:
  Ptr<models::ModelBase> builder_;
  Ptr<ExpressionGraph> graph_;
  Ptr<ExpressionGraph> graphAvg_;

  void execute(Ptr<data::Batch> batch);

public:
  SingletonGraph(Ptr<Config> config)
      : GraphGroup(config),
        ExponentialSmoothing(options_->get<float>("exponential-smoothing")) {
    auto deviceId = options_->getDevices()[0];
    graph_ = New<ExpressionGraph>();
    graph_->setDevice(deviceId);
    graph_->getBackend()->setClip(options_->get<float>("clip-gemm"));
    graph_->reserveWorkspaceMB(options_->get<size_t>("workspace"));
    opt_ = Optimizer(options_);
    builder_ = models::from_config(options_, models::usage::training);
  }

  void update(Ptr<data::Batch> batch) override {
    ABORT_IF(finalized_, "Training has already finished.");
    execute(batch);
  }

  void load() override {
    if(!options_->get<bool>("no-reload")) {
      std::string name = options_->get<std::string>("model");

      if(boost::filesystem::exists(name)) {
        if(scheduler_)
          scheduler_->load(name);

        if(mvAvg_ && boost::filesystem::exists(name + ".orig.npz")) {
          // Load the original parameters from model.npz
          builder_->load(graph_, name + ".orig.npz");

          // Load the averaged parameters from model.npz
          graphAvg_ = New<ExpressionGraph>();
          graphAvg_->setDevice(graph_->getDeviceId());
          builder_->load(graphAvg_, name);
          graphAvg_->forward();
        } else {
          builder_->load(graph_, name);
        }

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

  void save(bool final = false) override {
    auto saveGraph = graph_;
    if(mvAvg_) {
      // The model with averaged parameters will be saved into model.npz as
      // it's a model which should be used for decoding
      saveGraph = graphAvg_;
      // Save the original parameters in model.npz.orig.npz
      std::string name = options_->get<std::string>("model");
      builder_->save(graph_, name + ".orig.npz");
    }

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

  virtual void finalize() override { finalized_ = true; }
};
}  // namespace marian
