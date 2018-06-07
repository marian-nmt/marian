#pragma once

#include <thread>

#include "3rd_party/threadpool.h"
#include "training/graph_group.h"

namespace marian {

class SyncGraphGroup : public GraphGroup {
public:
  virtual void setScheduler(Ptr<Scheduler> scheduler);

private:
  std::vector<Ptr<models::ModelBase>> builders_;
  std::vector<Ptr<ExpressionGraph>> graphs_;
  std::vector<DeviceId> devices_;

  std::vector<Tensor> params_;
  std::vector<Tensor> grads_;
  std::vector<Tensor> tmpTensors_;
  std::vector<Ptr<TensorAllocator>> paramsAllocs_;

  std::vector<Ptr<OptimizerBase>> shardOpt_;

  int shardSize_;
  bool first_{true};

  std::vector<Tensor> paramsAvg_;
  std::vector<Ptr<TensorAllocator>> paramsAllocAvg_;
  bool movingAvg_{false};
  float mvDecay_{1e-4};
  size_t delay_{1};

  void updateMovingAverage(Tensor paramsAvg, Tensor params, size_t batches);

  void fetchParams(Tensor oldParams, const std::vector<Tensor>& params);

  void execute(Ptr<data::Batch> batch);

public:
  SyncGraphGroup(Ptr<Config> config)
      : GraphGroup(config),
        devices_{options_->getDevices()},
        movingAvg_{options_->get<float>("exponential-smoothing") > 0},
        mvDecay_{options_->get<float>("exponential-smoothing")},
        delay_{options_->get<size_t>("optimizer-delay")} {
    for(auto device : devices_) {
      auto graph = New<ExpressionGraph>();
      graph->setDevice(device);
      graph->reserveWorkspaceMB(options_->get<size_t>("workspace"));
      graph->getBackend()->setClip(options_->get<float>("clip-gemm"));

      graphs_.push_back(graph);
      shardOpt_.push_back(Optimizer(options_));
      builders_.push_back(models::from_config(options_, models::usage::training));
    }
  }

  void update(Ptr<data::Batch> batch) {
    ABORT_IF(finalized_, "Training has already finished.");
    execute(batch);
  }

  void load() {
    if(!options_->get<bool>("no-reload")) {
      std::string name = options_->get<std::string>("model");

      if(boost::filesystem::exists(name)) {
        size_t i = 0;
        if(scheduler_)
          scheduler_->load(name);
        for(auto graph : graphs_)
          builders_[i++]->load(graph, name);

        // @TODO: probably we want to have the list of DeviceIds as an attribute
        std::vector<Ptr<Backend>> backends;
        for(auto graph : graphs_)
          backends.push_back(graph->getBackend());
        shardOpt_[0]->load(name + ".optimizer.npz", shardOpt_, backends);

      } else if(options_->has("pretrained-model")) {
        std::string init = options_->get<std::string>("pretrained-model");
        LOG(info,
            "Initialize model weights with the pre-trained model {}",
            init);
        size_t i = 0;
        for(auto graph : graphs_)
          builders_[i++]->load(graph, init, false);
      }
    }
  }

  void save(bool final = false) {
    if(final && scheduler_) {
      if(movingAvg_ && paramsAvg_.size() > 0)
        for(auto graph : graphs_)
          fetchParams(graph->params()->vals(), paramsAvg_);

      scheduler_->validate(graphs_, true);

      if(movingAvg_ && paramsAvg_.size() > 0)
        for(auto graph : graphs_)
          fetchParams(graph->params()->vals(), params_);
    }

    save(graphs_[0], final);
  }

  void save(Ptr<ExpressionGraph> graph, bool final = false) {
    int idx = 0;
    for(int i = 0; i < graphs_.size(); ++i) {
      if(graph == graphs_[i]) {
        idx = i;
        break;
      }
    }

    if(movingAvg_ && paramsAvg_.size() > 0)
      fetchParams(graphs_[idx]->params()->vals(), paramsAvg_);

    std::string name = options_->get<std::string>("model");

    if(options_->get<bool>("overwrite")) {
      builders_[idx]->save(graphs_[idx], name, true);
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
        builders_[idx]->save(graphs_[idx], nameOverwrite);
      }

      builders_[idx]->save(graphs_[idx], name, true);
      if(scheduler_)
        scheduler_->save(name);
    }

    if(movingAvg_ && paramsAvg_.size() > 0)
      fetchParams(graphs_[idx]->params()->vals(), params_);

    size_t totalSize = graphs_[idx]->params()->vals()->size();
    shardOpt_[idx]->save(name + ".optimizer.npz", shardOpt_, totalSize);
  }

  Ptr<data::BatchStats> collectStats() {
    return GraphGroup::collectStats(graphs_[0], builders_[0], devices_.size() * delay_);
  }

  virtual void finalize() {
    finalized_ = true;
  }
};
}
