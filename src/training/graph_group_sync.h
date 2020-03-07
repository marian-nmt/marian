#pragma once

#include "training/graph_group.h"
#include "training/communicator.h"
#include "training/exponential_smoothing.h"

namespace marian {

class SyncGraphGroup : public GraphGroup, public ExponentialSmoothing {
  using Base = GraphGroup;
  const double delay_{1.}; // optimizer-delay parameter. Fractional means to use a fraction of whatever the MB size is

  Ptr<ICommunicator> comm_; // [not null] communicator, e.g. NCCLCommunicator
  Ptr<IMPIWrapper> mpi_;    // [not null] all MPI-like communication goes through this (this is a dummy implementation if no MPI run)

  std::vector<DeviceId> devices_;                         // [deviceIndex]
  std::vector<Ptr<models::ICriterionFunction>> builders_; // [deviceIndex]
  std::vector<Ptr<ExpressionGraph>> graphs_;              // [deviceIndex]

  std::vector<Ptr<OptimizerBase>> shardOpt_;       // [deviceIndex]
  std::vector<Tensor> paramsAvg_;                  // [deviceIndex] exponentially smoothed parameters, sharded
  // @TODO: instead, create an array of ExponentialSmoothing objects, and don't use ExponentialSmoothing as a base class
  std::vector<Ptr<TensorAllocator>> paramsAllocs_; // [deviceIndex] we must hold a reference to the memory until this class dies
  // @TODO: move this nto ExponentialSmoothing, together with paramsAvg_?

  // state for update()
  bool first_{ true };                           // gets interpreted and cleared by update()
  std::vector<Ptr<data::Batch>> pendingBatches_; // in case of dynamic MB-size scaling, we temporarly buffer up batches across update() calls until enough
  double updateMultiplier_{1};                  // multiplier not applied in collectStats() (no multiplier if not mini-batch-fit)

  void initialize(const Ptr<data::Batch>& exampleBatch);
  void initializeAvg();

  bool isMainProcess() const { return mpi_->myMPIRank() == 0; } // (we need this test a few times)
  void barrier() const { mpi_->barrier(); } // (we need this several times)
  void swapParamsAvg() { if (mvAvg_ && paramsAvg_.size() > 0) comm_->swapParams(paramsAvg_); } // note: must call this on all MPI ranks in parallel

  bool tryGetSubBatches(Ptr<data::Batch> newBatch, std::vector<Ptr<data::Batch>>& subBatches, size_t& numReadBatches);
  void update(std::vector<Ptr<data::Batch>> subBatches, size_t numReadBatches);

public:
  SyncGraphGroup(Ptr<Options> config, Ptr<IMPIWrapper> mpi);

  void setScheduler(Ptr<Scheduler> scheduler) override;

  void update(Ptr<data::Batch> batch) override;

  void load() override;
  void save(bool final = false) override;

  Ptr<data::BatchStats> collectStats(const std::vector<Ptr<Vocab>>&);
  void finalize() override;

  // @TODO: consider to make this a virtual as well? Currently it is a template dispatch
};
}  // namespace marian
