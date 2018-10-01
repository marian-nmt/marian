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
  Ptr<ICommunicator> comm_;

  Ptr<IMPIWrapper> mpi_; // multi-node only; all MPI-like communication goes through this

  std::vector<DeviceId> devices_;                // [deviceIndex]
  std::vector<Ptr<models::ModelBase>> builders_; // [deviceIndex]
  std::vector<Ptr<ExpressionGraph>> graphs_;     // [deviceIndex]

  std::vector<Ptr<OptimizerBase>> shardOpt_;     // [deviceIndex]

  int shardSize_; // @TODO: what is this?
  bool first_{true};

  std::vector<Tensor> paramsAvg_;
  std::vector<Ptr<TensorAllocator>> paramsAllocs_;

  size_t delay_{1};

  void initialize(const Ptr<data::Batch>& exampleBatch);
  void initializeAvg();

public:
  SyncGraphGroup(Ptr<Config> config);

  void update(Ptr<data::Batch> batch) override;

  void load() override;
  void save(bool final = false) override;

  Ptr<data::BatchStats> collectStats() {
    return GraphGroup::collectStats(graphs_[0], builders_[0], numBatches());
  }

  // @TODO: do we need to check mpi_ for null?
  size_t numBatches() { return devices_.size() * (mpi_ ? mpi_->numMPIProcesses() : 1) * delay_; }

  virtual void finalize() override { finalized_ = true; }
};
}  // namespace marian
