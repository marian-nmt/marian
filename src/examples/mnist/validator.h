#pragma once

#include "common/options.h"
#include "data/batch_generator.h"
#include "graph/expression_graph.h"
#include "models/model_base.h"
#include "training/validator.h"

#include "examples/mnist/dataset.h"

using namespace marian;

namespace marian {

class MNISTAccuracyValidator : public Validator<data::MNISTData, models::IModel> {
public:
  MNISTAccuracyValidator(Ptr<Options> options) : Validator(std::vector<Ptr<Vocab>>(), options, false) {
    createBatchGenerator(/*isTranslating=*/false);
    builder_ = models::createModelFromOptions(options, models::usage::translation);
  }

  virtual ~MNISTAccuracyValidator(){}

  virtual void keepBest(const std::vector<Ptr<ExpressionGraph>>& /*graphs*/) override {
    LOG(warn, "Keeping best model for MNIST examples is not supported");
  }

  std::string type() override { return "accuracy"; }

protected:
  virtual float validateBG(const std::vector<Ptr<ExpressionGraph>>& graphs) override {
    float correct = 0;
    size_t samples = 0;

    for(auto batch : *batchGenerator_) {
      auto probs = builder_->build(graphs[0], batch, true).getLogits();
      graphs[0]->forward();

      std::vector<float> scores;
      probs->val()->get(scores);

      correct += countCorrect(scores, batch->labels());
      samples += batch->size();
    }

    return correct / float(samples);
  }

private:
  float countCorrect(const std::vector<float>& probs, const std::vector<float>& labels) {
    size_t numLabels = probs.size() / labels.size();
    float numCorrect = 0;
    for(size_t i = 0; i < probs.size(); i += numLabels) {
      auto pred = std::distance(probs.begin() + i,
                                std::max_element(probs.begin() + i, probs.begin() + i + numLabels));
      if(pred == labels[i / numLabels])
        ++numCorrect;
    }
    return numCorrect;
  }
};

}  // namespace marian
