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
<<<<<<< HEAD
                                     double multiplier = 1.) {
    auto stats = New<data::BatchStats>();

    size_t numFiles = options_->get<bool>("tsv", false)
                          ? options_->get<size_t>("tsv-fields")
                          : options_->get<std::vector<std::string>>("train-sets").size();

    // Initialize first batch to step size
    size_t first = options_->get<size_t>("mini-batch-fit-step");

    // Increase batch size and sentence length by this step size
    size_t step = options_->get<size_t>("mini-batch-fit-step");

    size_t maxLength = options_->get<size_t>("max-length");
    maxLength = (size_t)(std::ceil(maxLength / (float)step) * step);

    // this should be only one class label per line on input, hence restricting length to 1
    std::vector<size_t> localMaxes(numFiles, maxLength);
    auto inputTypes = options_->get<std::vector<std::string>>("input-types", {});
    for(int i = 0; i < inputTypes.size(); ++i)
      if(inputTypes[i] == "class")
        localMaxes[i] = 1;

    size_t maxBatch = 512;
    bool fits = true;
    while(fits) {
      std::vector<size_t> lengths(numFiles, first);
      for(int j = 0; j < lengths.size(); ++j) // apply length restrictions
        lengths[j] = std::min(lengths[j], localMaxes[j]);

      auto batch = data::CorpusBatch::fakeBatch(lengths, vocabs, maxBatch, options_);
      auto cost = model->build(graph, batch);
      fits = graph->fits();
      if(fits)
        maxBatch *= 2;
    }

    // Do a binary search for maxmimum batch size that fits into given workspace memory
    // for a tested sentence length.
    for(size_t i = step; i <= maxLength; i += step) {
      size_t start = 1;
      size_t end = maxBatch;

      std::vector<size_t> lengths(numFiles, i);
      for(int j = 0; j < lengths.size(); ++j)  // apply length restrictions
        lengths[j] = std::min(lengths[j], localMaxes[j]);
      fits = true;

      do {
        size_t current = (start + end) / 2;
        auto batch = data::CorpusBatch::fakeBatch(lengths, vocabs, current, options_);
        auto cost = model->build(graph, batch);
        fits = graph->fits();

        LOG(debug, "[batching] length: {} - size: {} - fits: {}", lengths[0], current, fits);

        if(fits) {
          stats->add(batch, multiplier);
          start = current + 1;
        } else {
          end = current - 1;
        }
      } while(end - start > step);

      maxBatch = start;
    }
    return stats;
  }

  void setTypicalTrgBatchWords(size_t typicalTrgBatchWords) { // needed for dynamic MB scaling
    typicalTrgBatchWords_ = typicalTrgBatchWords;
  }
};

/**
 *  Base class for multi-node versions of GraphGroups.
 */
class MultiNodeGraphGroupBase : public GraphGroup {
  using Base = GraphGroup;

protected:
  Ptr<IMPIWrapper> mpi_; // all MPI-like communication goes through this
=======
                                     double multiplier = 1.);
>>>>>>> master

  void setTypicalTrgBatchWords(size_t typicalTrgBatchWords);
};

}  // namespace marian
