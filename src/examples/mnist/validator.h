#pragma once

#include "data/batch_generator.h"
#include "graph/expression_graph.h"
#include "training/config.h"
#include "training/validator.h"

#include "examples/mnist/mnist.h"


using namespace marian;

namespace marian {

template <class Builder>
class AccuracyValidator : public Validator<data::MNIST> {
  private:
    Ptr<Builder> builder_;

  public:
    template <class ...Args>
    AccuracyValidator(std::vector<Ptr<Vocab>> vocabs,
                      Ptr<Config> options,
                      Args ...args)
     : Validator(vocabs, options),
       builder_(New<Builder>(options, keywords::inference=true, args...)) {
      initLastBest();
    }

    virtual float validateBG(Ptr<ExpressionGraph> graph,
                             Ptr<data::BatchGenerator<data::MNIST>> batchGenerator) {
      float cor = 0;
      size_t samples = 0;

      while(*batchGenerator) {
        auto batch = batchGenerator->next();
        auto probs = builder_->build(graph, batch);
        graph->forward();

        std::vector<float> scores;
        probs->val()->get(scores);

        cor += countCorrect(scores, batch->inputs()[1].data());
        samples += batch->size();
      }

      return cor / float(samples);
    }

    virtual void keepBest(Ptr<ExpressionGraph> graph) {
      // not supported
    }

    bool lowerIsBetter() {
      return false;
    }

    std::string type() { return "accuracy"; }

  private:

    float countCorrect(const std::vector<float>& probs, const std::vector<float>& labels) {
      size_t numLabels = probs.size() / labels.size();
      float numCorrect = 0;
      for (size_t i = 0; i < probs.size(); i += numLabels) {
        auto pred = std::distance(probs.begin() + i,
                                  std::max_element(probs.begin() + i,
                                                   probs.begin() + i + numLabels));
        if (pred == labels[i / numLabels])
          ++numCorrect;
      }
      return numCorrect;
    }
};


}
