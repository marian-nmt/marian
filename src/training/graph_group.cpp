#include "training/graph_group.h"

namespace marian {

GraphGroup::GraphGroup(Ptr<Options> options) : options_(options), opt_(Optimizer(options)) {}

void GraphGroup::validate() {
  ABORT_IF(finalized_, "Training has already finished.");
}

void GraphGroup::finalize() {
  finalized_ = true;
}

Ptr<data::BatchStats> GraphGroup::collectStats(Ptr<ExpressionGraph> graph,
                                               Ptr<models::ICriterionFunction> model,
                                               const std::vector<Ptr<Vocab>>& vocabs,
                                               double multiplier) {
  auto stats = New<data::BatchStats>();
  size_t numFiles = numberOfInputFiles();

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
    } while(end - start > step); // @TODO: better replace with `end >= start` to remove the step here

    maxBatch = start;
  }
  return stats;
}

void GraphGroup::setTypicalTrgBatchWords(size_t typicalTrgBatchWords) { // needed for dynamic MB scaling
  typicalTrgBatchWords_ = typicalTrgBatchWords;
}

size_t GraphGroup::numberOfInputFiles() {
  if(options_->get<bool>("tsv", false)) {
    size_t n = options_->get<size_t>("tsv-fields");
    if(n > 0 && options_->get("guided-alignment", std::string("none")) != "none")
      --n;
    if(n > 0 && options_->hasAndNotEmpty("data-weighting"))
      --n;
    return n;
  }
  return options_->get<std::vector<std::string>>("train-sets").size();
}

}  // namespace marian
