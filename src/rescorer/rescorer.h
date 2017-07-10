#pragma once

#include "marian.h"

#include "common/config.h"
#include "data/batch_generator.h"
#include "data/corpus.h"
#include "graph/expression_graph.h"
#include "models/amun.h"
#include "models/model_task.h"
#include "models/s2s.h"
#include "rescorer/score_collector.h"

namespace marian {

using namespace data;

template <class Builder>
class Rescorer {
private:
  Ptr<Builder> builder_;

public:
  template <typename ...Args>
  Rescorer(Args ...args) :
    builder_(new Builder(args...)) {}

  void load(Ptr<ExpressionGraph> graph, const std::string& modelFile) {
    builder_->load(graph, modelFile);
  }

  Expr buildToScore(Ptr<ExpressionGraph> graph, Ptr<data::CorpusBatch> batch) {
    return builder_->buildToScore(graph, batch);
  }
};

template <class Model>
class Rescore : public ModelTask {
private:
  Ptr<Config> options_;
  Ptr<Corpus> corpus_;
  Ptr<ExpressionGraph> graph_;
  Ptr<Model> model_;

public:
  Rescore(Ptr<Config> options)
      : options_(options),
        corpus_(New<Corpus>(options_)),
        graph_(New<ExpressionGraph>(true)) {
    corpus_->prepare();

    auto device = options_->get<std::vector<size_t>>("devices").front();
    graph_->setDevice(device);
    graph_->reserveWorkspaceMB(options_->get<size_t>("workspace"));

    auto modelFile = options_->get<std::string>("model");
    auto modelOptions = New<Config>(*options);
    try {
      modelOptions->loadModelParameters(modelFile);
    } catch(std::runtime_error& e) {
      LOG(warn)->warn("No model settings found in model file");
    }
    model_ = New<Model>(modelOptions, keywords::inference = true);
    model_->load(graph_, modelFile);
  }

  void run() {
    Ptr<BatchGenerator<Corpus>> batchGenerator
        = New<BatchGenerator<Corpus>>(corpus_, options_);
    batchGenerator->prepare(false);

    auto output = New<ScoreCollector>();

    while(*batchGenerator) {
      auto batch = batchGenerator->next();

      auto costNode = model_->buildToScore(graph_, batch);
      graph_->forward();

      std::vector<float> scores;
      costNode->val()->get(scores);

      for(size_t i = 0; i < batch->size(); ++i) {
        output->Write(batch->getSentenceIds()[i], scores[i]);
      }
    }
  }
};

}
