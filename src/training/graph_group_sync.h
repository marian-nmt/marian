#pragma once

#include <thread>

#include "3rd_party/threadpool.h"
#include "training/communicator.h"
#include "training/exponential_smoothing.h"
#include "training/graph_group.h"

namespace marian {

class SyncGraphGroup : public GraphGroup, public ExponentialSmoothing {
public:
  virtual void setScheduler(Ptr<Scheduler> scheduler) override;

private:
  Ptr<Communicator> comm_;

  std::vector<Ptr<models::ModelBase>> builders_;
  std::vector<Ptr<ExpressionGraph>> graphs_;
  std::vector<DeviceId> devices_;

  std::vector<Ptr<OptimizerBase>> shardOpt_;

  int shardSize_;
  bool first_{true};

  std::vector<Tensor> paramsAvg_;
  std::vector<Ptr<TensorAllocator>> paramsAllocs_;

  size_t delay_{1};

  void initialize(const std::vector<Ptr<data::Batch>>& batches);
  void initializeAvg();

  void execute(Ptr<data::Batch> batch);

public:
  SyncGraphGroup(Ptr<Config> config);

  void update(Ptr<data::Batch> batch) override {
    ABORT_IF(finalized_, "Training has already finished.");
    execute(batch);
  }

  void load() override;
  void save(bool final = false) override;
  void save(Ptr<ExpressionGraph> graph, bool final = false);

  Ptr<data::BatchStats> collectStats() {
    return GraphGroup::collectStats(graphs_[0], builders_[0], numBatches());
  }

  size_t numBatches() { return devices_.size() * delay_; }

  virtual void finalize() override { finalized_ = true; }
};
}  // namespace marian
