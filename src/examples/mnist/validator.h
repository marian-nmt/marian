#pragma once

#include "common/config.h"
#include "data/batch_generator.h"
#include "graph/expression_graph.h"
#include "training/validator.h"

#include "examples/mnist/dataset.h"

using namespace marian;

namespace marian {

template <class Builder>
class AccuracyValidator : public Validator<data::MNIST> {
private:
  Ptr<Builder> builder_;

public:
  template <class... Args>
  AccuracyValidator(Ptr<Config> options, Args... args)
      : Validator(std::vector<Ptr<Vocab>>(), options),
        builder_(New<Builder>(options, keywords::inference = true, args...)) {
    initLastBest();
  }

  virtual void keepBest(Ptr<ExpressionGraph> graph) {
    auto model = options_->get<std::string>("model");
    builder_->save(graph, model + ".best-" + type() + ".npz", true);
  }

  bool lowerIsBetter() { return false; }

  std::string type() { return "accuracy"; }

protected:
  virtual float validateBG(
      Ptr<ExpressionGraph> graph,
      Ptr<data::BatchGenerator<data::MNIST>> batchGenerator) {
    float correct = 0;
    size_t samples = 0;

    while(*batchGenerator) {
      auto batch = batchGenerator->next();
      auto probs = builder_->build(graph, batch);
      graph->forward();

      std::vector<float> scores;
      probs->val()->get(scores);

      correct += countCorrect(scores, batch->labels());
      samples += batch->size();
    }

    return correct / float(samples);
  }

private:
  float countCorrect(const std::vector<float>& probs,
                     const std::vector<float>& labels) {
    size_t numLabels = probs.size() / labels.size();
    float numCorrect = 0;
    for(size_t i = 0; i < probs.size(); i += numLabels) {
      auto pred = std::distance(
          probs.begin() + i,
          std::max_element(probs.begin() + i, probs.begin() + i + numLabels));
      if(pred == labels[i / numLabels])
        ++numCorrect;
    }
    return numCorrect;
  }
};
}
