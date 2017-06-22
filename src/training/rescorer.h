#pragma once

#include "marian.h"

#include "data/batch_generator.h"
#include "data/corpus.h"
#include "graph/expression_graph.h"
#include "models/amun.h"
#include "models/model_task.h"
#include "models/s2s.h"
#include "training/config.h"

namespace marian {

using namespace data;

template <class Model>
class Rescorer : public ModelTask {
private:
  Ptr<Config> options_;

public:
  Rescorer(Ptr<Config> options) : options_(options) {}

  void run() {
    auto corpus = New<data::Corpus>(options_);
    corpus->prepare();

    // @TODO: this is a bug that no default device is set
    auto graph = New<ExpressionGraph>();
    auto device = options_->get<std::vector<size_t>>("devices").front();
    graph->setDevice(device);

    auto model = New<Model>(options_);
    model->load(graph, options_->get<std::string>("model"));

    Ptr<BatchGenerator<Corpus>> batchGenerator
        = New<BatchGenerator<Corpus>>(corpus, options_);

    // @TODO: a temporal fix as the order of sentences in a batch is random
    batchGenerator->forceBatchSize(1);
    batchGenerator->prepare(false);

    while(*batchGenerator) {
      auto batch = batchGenerator->next();
      auto costNode = model->buildToScore(graph, batch);
      graph->forward();

      std::vector<float> scores;
      costNode->val()->get(scores);

      for(auto score : scores)
        std::cout << score << std::endl;
    }
  }
};

Ptr<ModelTask> rescorerByType(Ptr<Config> options) {
  std::string type = options->get<std::string>("type");

  if(type == "s2s") {
    return New<Rescorer<S2S>>(options);
  } else if(type == "amun") {
    return New<Rescorer<Amun>>(options);
  } else {
    UTIL_THROW2("Unknown model type: " + type);
  }
}
}
