#pragma once

#include "training/graph_group.h"
#include "common/filesystem.h"
#include "training/exponential_smoothing.h"

#include <future>

namespace marian {

/**
 * Single GPU training
 */
class SingletonGraph : public GraphGroup, public ExponentialSmoothing {
public:
  virtual void setScheduler(Ptr<Scheduler> scheduler) override;

private:
  Ptr<models::ICriterionFunction> builder_;
  Ptr<ExpressionGraph> graph_;
  Ptr<ExpressionGraph> graphAvg_;

  void execute(Ptr<data::Batch> batch);

public:
  SingletonGraph(Ptr<Options> config, Ptr<IMPIWrapper> mpi)
      : GraphGroup(config),
        ExponentialSmoothing(config) {
    ABORT_IF(mpi->numMPIProcesses() != 1, "SingletonGraph does not support multiple MPI processes");
    // Get device ID
    auto devices = Config::getDevices(options_);
    ABORT_IF(devices.size() != 1, "Only one device ID should be provided for singleton training");
    auto deviceId = devices[0];
    // Initialize graph
    graph_ = New<ExpressionGraph>();
    graph_->setDevice(deviceId);
    graph_->setCheckpointing(options_->get<bool>("gradient-checkpointing"));
    graph_->reserveWorkspaceMB(options_->get<size_t>("workspace"));
    opt_ = Optimizer(options_);
    builder_ = models::createCriterionFunctionFromOptions(options_, models::usage::training);
  }

  void update(Ptr<data::Batch> batch) override {
    validate();
    execute(batch);
  }

  void load() override {
    if(!options_->get<bool>("no-reload")) {
      std::string name = options_->get<std::string>("model");

      if(filesystem::exists(name)) {
        if(scheduler_)
          scheduler_->load(name);

        if(mvAvg_ && filesystem::exists(name + ".orig.npz")) {
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

        opt_->load(name + ".optimizer.npz", {opt_}, {graph_->getBackend()},
          /*scatterStateFn=*/[&](const std::vector<float>& data, const OptimizerBase::ScatterStateSetFunc& setFn) {
            setFn(/*localDeviceIndex=*/0, data.begin(), data.end());
          });
      } else if(options_->hasAndNotEmpty("pretrained-model")) {
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

    opt_->save(name + ".optimizer.npz", {opt_},
      /*gatherStateFn=*/[&](const OptimizerBase::GatherStateGetFunc& getFn) {
        return getFn(/*localDeviceIndex=*/0);
      });
  }

  Ptr<data::BatchStats> collectStats(const std::vector<Ptr<Vocab>>& vocabs) {
    return GraphGroup::collectStats(graph_, builder_, vocabs);
  }

  virtual void finalize() override { finalized_ = true; }
};
}  // namespace marian
