#pragma once

#include "common/definitions.h"
#include "data/batch_generator.h"
#include "models/model_base.h"
#include "optimizers/optimizers.h"
#include "training/scheduler.h"
#include "graph/expression_graph.h"

namespace marian {

class GraphGroup {
protected:
  Ptr<Config> options_;
  Ptr<OptimizerBase> opt_;
  Ptr<Scheduler> scheduler_;
  bool finalized_{false};

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

  virtual void finalize() = 0;

  virtual void setScheduler(Ptr<Scheduler> scheduler) = 0;

  virtual Ptr<data::BatchStats> collectStats() = 0;

  virtual Ptr<data::BatchStats> collectStats(Ptr<ExpressionGraph> graph,
                                             Ptr<models::ModelBase> model,
                                             size_t multiplier = 1) {
    auto stats = New<data::BatchStats>();

    size_t numFiles = options_->get<std::vector<std::string>>("train-sets").size();

    size_t first = options_->get<size_t>("mini-batch-fit-step");
    size_t step = options_->get<size_t>("mini-batch-fit-step");

    size_t maxLength = options_->get<size_t>("max-length");
    maxLength = std::ceil(maxLength / (float)step) * step;

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
      bool fits = true;

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
}
