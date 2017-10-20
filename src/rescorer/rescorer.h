#pragma once

#include "marian.h"

#include "common/config.h"
#include "data/batch_generator.h"
#include "data/corpus.h"
#include "models/model_task.h"
#include "rescorer/score_collector.h"
#include "training/scheduler.h"
#include "training/validator.h"

namespace marian {

using namespace data;

class Rescorer {
private:
  Ptr<models::ModelBase> builder_;

public:
  Rescorer(Ptr<Options> options) : builder_(models::from_options(options)) {}

  void load(Ptr<ExpressionGraph> graph, const std::string& modelFile) {
    builder_->load(graph, modelFile);
  }

  Expr build(Ptr<ExpressionGraph> graph, Ptr<data::CorpusBatch> batch) {
    return builder_->build(graph, batch);
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
      LOG(warn, "No model settings found in model file");
    }

    Ptr<Options> temp = New<Options>();
    temp->merge(options);
    temp->set("inference", true);
    temp->set("cost-type", "ce-rescore");
    model_ = New<Model>(temp);

    model_->load(graph_, modelFile);
  }

  void run() {
    LOG(info, "Scoring");

    auto batchGenerator = New<BatchGenerator<Corpus>>(corpus_, options_);
    batchGenerator->prepare(false);

    auto output = New<ScoreCollector>();

    bool summarize = options_->has("summary");
    std::string summary
        = summarize ? options_->get<std::string>("summary") : "cross-entropy";

    float sumCost = 0;
    size_t sumWords = 0;
    size_t sumSamples = 0;

    while(*batchGenerator) {
      auto batch = batchGenerator->next();

      auto costNode = model_->build(graph_, batch);
      graph_->forward();

      std::vector<float> scores;
      costNode->val()->get(scores);

      for(auto s : scores)
        sumCost += s;
      sumWords += batch->back()->batchWords();
      sumSamples += batch->size();

      if(!summarize) {
        for(size_t i = 0; i < batch->size(); ++i) {
          output->Write(batch->getSentenceIds()[i], scores[i]);
        }
      }
    }

    if(summarize) {
      float cost = 0;
      if(summary == "perplexity")
        cost = std::exp(-(float)sumCost / (float)sumWords);
      else if(summary == "ce-sum")
        cost = -sumCost;
      else if(summary == "ce-mean-words")
        cost = -(float)sumCost / (float)sumWords;
      else
        cost = -sumCost / sumSamples;

      LOG(info, "Reporting {} summary", summary);
      std::cout << cost << std::endl;
    }
  }
};
}
