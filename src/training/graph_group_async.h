#pragma once

#include <condition_variable>
#include <future>
#include <thread>

#include <boost/filesystem.hpp>
#include <boost/thread/locks.hpp>
#include <boost/thread/shared_mutex.hpp>

#include "3rd_party/threadpool.h"
#include "training/exponential_smoothing.h"
#include "training/graph_group.h"

namespace marian {

class AsyncGraphGroup : public GraphGroup, public ExponentialSmoothing {
public:
  virtual void setScheduler(Ptr<Scheduler> scheduler) override;

protected:
  bool first_{true};

  std::vector<Ptr<models::ModelBase>> builders_;
  std::vector<Ptr<ExpressionGraph>> graphs_;
  std::vector<DeviceId> devices_;

  std::mutex sync_;
  std::vector<std::mutex> shardSync_;

  std::mutex schedulerMutex_;

  std::vector<Tensor> params_;
  std::vector<Ptr<TensorAllocator>> paramsAlloc_;

  std::vector<Tensor> grads_;
  std::vector<Ptr<TensorAllocator>> gradsAlloc_;

  std::vector<Ptr<OptimizerBase>> shardOpt_;

  int shardSize_;

  std::vector<Tensor> paramsAvg_;
  std::vector<Ptr<TensorAllocator>> paramsAllocAvg_;
  std::unique_ptr<ThreadPool> pool_;

  size_t optimizerDelay_{1};

  virtual void fetchParams(Tensor oldParams,
                           const std::vector<Tensor>& params,
                           int device_id);

  virtual void pushGradients(Tensor newGrads,
                             size_t batch_words,
                             int device_id);

  virtual void init(Ptr<data::Batch> batch);
  void execute(Ptr<data::Batch> batch);

public:
  AsyncGraphGroup(Ptr<Config> config);

  void update(Ptr<data::Batch> batch) override {
    ABORT_IF(finalized_, "Training has already finished.");
    execute(batch);
  }

  void load() override;
  void save(bool final = false) override;
  void save(Ptr<ExpressionGraph>, bool final = false);

  Ptr<data::BatchStats> collectStats() {
    return GraphGroup::collectStats(graphs_[0], builders_[0]);
  }

  virtual void finalize() override;
};

}  // namespace marian
