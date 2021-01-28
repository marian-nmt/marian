#pragma once

#include "3rd_party/threadpool.h"
#include "training/graph_group.h"

#include <future>
#include <thread>

namespace marian {

class AsyncGraphGroup : public GraphGroup {
public:
  virtual void setScheduler(Ptr<Scheduler> scheduler) override;

protected:
  bool first_{true};

  std::mutex sync_;
  std::vector<std::mutex> shardSync_;

  std::mutex schedulerMutex_;

  std::vector<Tensor> params_;
  std::vector<Ptr<TensorAllocator>> paramsAlloc_;

  std::vector<Tensor> grads_;
  std::vector<Ptr<TensorAllocator>> gradsAlloc_;

  int shardSize_;

  std::unique_ptr<ThreadPool> pool_;

  size_t optimizerDelay_{1};

  virtual void fetchParams(Tensor oldParams,
                           const std::vector<Tensor>& params,
                           int device_id);

  virtual void pushGradients(Tensor newGrads,
                             int device_id,
                             size_t mbSize);

  virtual void init(Ptr<data::Batch> batch);
  void execute(Ptr<data::Batch> batch);

public:
  AsyncGraphGroup(Ptr<Options> config, Ptr<IMPIWrapper> mpi);

  void update(Ptr<data::Batch> batch) override {
    validate();
    execute(batch);
  }

  // @TODO: give it a fake batch generator which own vocabs instead of passing vocabs
  Ptr<data::BatchStats> collectStats(const std::vector<Ptr<Vocab>>& vocabs) override {
    return GraphGroup::collectStats(graphs_[0], models_[0], vocabs);
  }

  virtual void finalize() override;
};

}  // namespace marian
