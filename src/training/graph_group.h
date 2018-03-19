#pragma once

#include "common/definitions.h"
#include "data/batch_generator.h"
#include "models/model_base.h"
#include "optimizers/optimizers.h"
#include "training/scheduler.h"

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

  virtual void wait(){};

  virtual void setScheduler(Ptr<Scheduler> scheduler) = 0;

  virtual Ptr<data::BatchStats> collectStats() = 0;
};
}
