#pragma once

#include "common/definitions.h"
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
  Ptr<Config> options_;
  Ptr<OptimizerBase> opt_;   // the optimizer
  Ptr<Scheduler> scheduler_; // scheduler that keeps track of how much has been processed
  bool finalized_{false};    // 'true' if training has completed (further updates are no longer allowed)

  Ptr<IMPIWrapper> mpi_; // all MPI-like communication goes through this

  bool scaleLearningRate_; // option "batch-flexible-lr"; "Scales the learning rate based on the number of words in a mini-batch"
  // @TODO: Is this the same as not averaging? On which level? Entire batch, or within-worker?
  float avgBatchWords_;    // option "batch-normal-words"; "Set number of words per batch that the learning rate corresponds to"

public:
  GraphGroup(Ptr<Config> options)
      : options_(options),
        opt_(Optimizer(options)),
        scaleLearningRate_(options->get<bool>("batch-flexible-lr")),
        avgBatchWords_(options->get<float>("batch-normal-words")) {}

  virtual ~GraphGroup() {}

  void setupMPI() {
    mpi_ = initMPI(/*multiThreaded=*/!options_->get<bool>("sync-sgd"));
  }

  /**
   * Setup MPI world size and rank of this node.
   */
  virtual void update(Ptr<data::Batch> batch) = 0;

  virtual void load() = 0;

  virtual void save(bool isFinal = false) = 0;

  virtual void finalize() {
    if (mpi_) {
      finalizeMPI(std::move(mpi_));
      ABORT_IF(mpi_, "MPI not finalized??");
    }
    finalized_ = true;
  }

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
  virtual Ptr<data::BatchStats> collectStats(Ptr<ExpressionGraph> graph,
                                             Ptr<models::ModelBase> model,
                                             size_t multiplier = 1) {
    auto stats = New<data::BatchStats>();

    size_t numFiles
        = options_->get<std::vector<std::string>>("train-sets").size();

    // Initialize first batch to step size
    size_t first = options_->get<size_t>("mini-batch-fit-step");

    // Increase batch size and sentence length by this step size
    size_t step = options_->get<size_t>("mini-batch-fit-step");

    size_t maxLength = options_->get<size_t>("max-length");
    maxLength = (size_t)(std::ceil(maxLength / (float)step) * step);

    // @TODO: ugly
    auto toptions = New<Options>();
    toptions->merge(options_);

    size_t maxBatch = 512;
    bool fits = true;
    while(fits) {
      std::vector<size_t> lengths(numFiles, first);
      auto batch = data::CorpusBatch::fakeBatch(lengths, maxBatch, toptions);
      auto cost = model->build(graph, batch);
      fits = graph->fits();
      if(fits)
        maxBatch *= 2;
    }

    for(size_t i = step; i <= maxLength; i += step) {
      size_t start = 1;
      size_t end = maxBatch;

      std::vector<size_t> lengths(numFiles, i);
      fits = true;

      do {
        size_t current = (start + end) / 2;
        auto batch = data::CorpusBatch::fakeBatch(lengths, current, toptions);
        auto cost = model->build(graph, batch);
        fits = graph->fits();

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
};
}  // namespace marian
