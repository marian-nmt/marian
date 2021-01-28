#pragma once

#include "training/graph_group.h"
#include "common/filesystem.h"

#include <future>

namespace marian {

/**
 * Single GPU training
 */
class SingletonGraph : public GraphGroup {
public:
  virtual void setScheduler(Ptr<Scheduler> scheduler) override;

private:
  void execute(Ptr<data::Batch> batch);

public:
  SingletonGraph(Ptr<Options> options, Ptr<IMPIWrapper> mpi)
      : GraphGroup(options, mpi) {

    LOG(warn, "This class only serves demonstration purposes. It should currently not be called from actual Marian code.");
    ABORT_IF(mpi->numMPIProcesses() != 1, "SingletonGraph does not support multiple MPI processes");
    ABORT_IF(devices_.size() != 1, "Only one device ID should be provided for singleton training");
  }

  void update(Ptr<data::Batch> batch) override {
    validate();
    execute(batch);
  }

  Ptr<data::BatchStats> collectStats(const std::vector<Ptr<Vocab>>& vocabs) override {
    return GraphGroup::collectStats(graphs_[0], models_[0], vocabs);
  }

  virtual void finalize() override { finalized_ = true; }
};
}  // namespace marian
