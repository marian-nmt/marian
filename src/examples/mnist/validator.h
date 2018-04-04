#pragma once

#include "common/config.h"
#include "data/batch_generator.h"
#include "graph/expression_graph.h"
#include "models/model_base.h"
#include "training/validator.h"

#include "examples/mnist/dataset.h"

using namespace marian;

namespace marian {

class AccuracyValidator : public Validator<data::MNISTData> {
public:
  AccuracyValidator(Ptr<Config> options)
      : Validator(std::vector<Ptr<Vocab>>(), options, false) {
    Ptr<Options> temp = New<Options>();
    temp->merge(options);
    temp->set("inference", true);
    builder_ = models::from_options(temp, models::usage::scoring);
  }

  virtual void keepBest(const std::vector<Ptr<ExpressionGraph>>& graphs) {
    LOG(warn, "Keeping best model for MNIST examples is not supported");
  }

  std::string type() { return "accuracy"; }

protected:
  virtual float validateBG(
      const std::vector<Ptr<ExpressionGraph>>& graphs,
      Ptr<data::BatchGenerator<data::MNISTData>> batchGenerator) {
    float correct = 0;
    size_t samples = 0;

    while(*batchGenerator) {
      auto batch = batchGenerator->next();
      auto probs = builder_->build(graphs[0], batch, true);
      graphs[0]->forward();

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
