#pragma once

#include <future>
#include <thread>

#include <boost/filesystem.hpp>
#include <boost/thread/locks.hpp>
#include <boost/thread/shared_mutex.hpp>

#include "3rd_party/threadpool.h"
#include "common/definitions.h"
#include "data/batch_generator.h"
#include "models/model_base.h"
#include "optimizers/optimizers.h"
#include "training/dropper.h"
#include "training/scheduler.h"
#include "training/sparse_tensor.h"
#include "training/training.h"
#include "training/validator.h"

namespace marian {

class GraphGroup {
protected:
  Ptr<Config> options_;
  Ptr<OptimizerBase> opt_;
  Ptr<Scheduler> scheduler_;

  bool scaleLearningRate_;
  float avgBatchWords_;

public:
  GraphGroup(Ptr<Config> options)
      : options_(options),
      opt_(Optimizer(options)),
      scaleLearningRate_(options->get<bool>("batch-flexible-lr")),
      avgBatchWords_(options->get<float>("batch-normal-words")) {}

  virtual ~GraphGroup() {}

  virtual void update(Ptr<data::Batch>) = 0;

  virtual void load() = 0;

  virtual void save(bool = false) = 0;

  virtual void setScheduler(Ptr<Scheduler> scheduler)  = 0;

  virtual Ptr<data::BatchStats> collectStats() = 0;
};

}
