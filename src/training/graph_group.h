#pragma once

#include "common/definitions.h"
#include "common/options.h"
#include "data/batch_generator.h"
#include "graph/expression_graph.h"
#include "models/model_base.h"
#include "optimizers/optimizers.h"
#include "training/scheduler.h"
#include "training/communicator.h"

namespace marian {

/**
 *  Base class for managing the training process across one, multiple gpus,
 *  or even multiple machines with multiple gpus.
 */
class GraphGroup {
protected:
  Ptr<Options> options_;
  Ptr<OptimizerBase> opt_;   // the optimizer

  Ptr<Scheduler> scheduler_; // scheduler that keeps track of how much has been processed

  bool finalized_{false};    // 'true' if training has completed (further updates are no longer allowed)
  size_t typicalTrgBatchWords_{ 0 }; // for dynamic batch sizing: typical batch size in words

public:
  GraphGroup(Ptr<Options> options);

  virtual ~GraphGroup() {}

  virtual void update(Ptr<data::Batch> batch) = 0;

  virtual void load() = 0;

  virtual void save(bool isFinal = false) = 0;

  void validate();

  virtual void finalize();

  virtual void setScheduler(Ptr<Scheduler> scheduler) = 0;

  /**
   * Determine maximal batch size that can fit into the given workspace
   * so that reallocation does not happen. Rather adjust the batch size
   * based on the stastistics collected here. Activated with
   * `--mini-batch-fit`.
   * In a multi-GPU scenario, the first GPU is used to determine the size.
   * The actual allowed size is then determined by multiplying it with the
   * number of devices, which is passed in as the 'multiplier'.
   */
  // @TODO: Can this be made const? It seems wrong to have a stateful method that still returns a result.
  Ptr<data::BatchStats> collectStats(Ptr<ExpressionGraph> graph,
                                     Ptr<models::ICriterionFunction> model,
                                     const std::vector<Ptr<Vocab>>& vocabs,
                                     double multiplier = 1.);

  void setTypicalTrgBatchWords(size_t typicalTrgBatchWords);
};

}  // namespace marian
