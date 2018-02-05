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
  std::vector<size_t> devices_;

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

  void updateMovingAverage(Tensor paramsAvg, Tensor params, size_t batches);

  void fetchParams(Tensor oldParams, const std::vector<Tensor>& params);

  void execute(Ptr<data::Batch> batch);

public:
  SyncGraphGroup(Ptr<Config> options)
      : GraphGroup(options),
        devices_{options_->get<std::vector<size_t>>("devices")},
        movingAvg_{options_->get<float>("exponential-smoothing") > 0},
        mvDecay_{options_->get<float>("exponential-smoothing")} {
    for(auto device : devices_) {
      auto graph = New<ExpressionGraph>();
      graph->setDevice(device);
      graph->reserveWorkspaceMB(options_->get<size_t>("workspace"));
      graphs_.push_back(graph);
      shardOpt_.push_back(Optimizer(options_));
      builders_.push_back(models::from_config(options_));
    }
  }

  void update(Ptr<data::Batch> batch) { execute(batch); }

  void load() {
    if(!options_->get<bool>("no-reload")) {
      std::string name = options_->get<std::string>("model");
      if(boost::filesystem::exists(name)) {
        size_t i = 0;
        if(scheduler_)
          scheduler_->load(name);
        for(auto graph : graphs_)
          builders_[i++]->load(graph, name);
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
    if(final && scheduler_)
      scheduler_->validate(graphs_, true);

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

    if(movingAvg_)
      fetchParams(graphs_[idx]->params()->vals(), paramsAvg_);

    if(options_->get<bool>("overwrite")) {
      std::string name = options_->get<std::string>("model");

      builders_[idx]->save(graphs_[idx], name, true);
      if(scheduler_)
        scheduler_->save(name);
    } else {
      std::string name = options_->get<std::string>("model");

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

    if(movingAvg_)
      fetchParams(graphs_[idx]->params()->vals(), params_);
  }

  Ptr<data::BatchStats> collectStats() {
    return builders_[0]->collectStats(graphs_[0], devices_.size());
  }
};
}
