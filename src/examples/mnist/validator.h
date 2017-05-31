#pragma once

#include <limits>
#include <cstdio>
#include <cstdlib>

#include "training/config.h"
#include "graph/expression_graph.h"
//#include "data/corpus.h"
//#include "data/batch_generator.h"

#include "translator/beam_search.h"
#include "translator/history.h"
#include "translator/printer.h"
#include "translator/output_collector.h"

#include "examples/mnist/mnist.h"
#include "examples/mnist/dataset.h"
#include "examples/mnist/batch_generator.h"


using namespace marian;
using namespace data;

namespace marian {

  class MNISTValidator {
    protected:
      Ptr<Config> options_;
      //std::vector<Ptr<Vocab>> vocabs_;
      float lastBest_;
      size_t stalled_{0};

    public:
      MNISTValidator(
          //std::vector<Ptr<Vocab>> vocabs,
                Ptr<Config> options)
       : options_(options),
         //vocabs_(vocabs),
         lastBest_{lowerIsBetter() ?
          std::numeric_limits<float>::max() :
          std::numeric_limits<float>::lowest() } {
      }

      virtual std::string type() = 0;

      virtual void keepBest(Ptr<ExpressionGraph> graph) = 0;

      virtual bool lowerIsBetter() {
        return true;
      }

      virtual void initLastBest() {
        lastBest_ = lowerIsBetter() ?
          std::numeric_limits<float>::max() :
          std::numeric_limits<float>::lowest();
      }

      size_t stalled() {
        return stalled_;
      }

      virtual float validate(Ptr<ExpressionGraph> graph) {
        using namespace data;
        //auto validPaths = options_->get<std::vector<std::string>>("valid-sets");
        auto corpus = New<MNIST>(
                   "../src/examples/mnist/t10k-images-idx3-ubyte",
                   "../src/examples/mnist/t10k-labels-idx1-ubyte");

        auto batchGenerator = New<MNISTBatchGenerator<MNIST>>(corpus, 200, 20);
        batchGenerator->prepare(false);

        float val = validateBG(graph, batchGenerator);

        if((lowerIsBetter() && lastBest_ > val) ||
           (!lowerIsBetter() && lastBest_ < val)) {
            stalled_ = 0;
            lastBest_ = val;
            //if(options_->get<bool>("keep-best"))
              //keepBest(graph);
        }
        else {
          stalled_++;
        }
        return val;
      };

      virtual float validateBG(Ptr<ExpressionGraph>,
                               Ptr<MNISTBatchGenerator<MNIST>>) = 0;

  };

  template <class Builder>
  class MNISTAccuracyValidator : public MNISTValidator {
    private:
      Ptr<Builder> builder_;

    public:
      template <class ...Args>
      MNISTAccuracyValidator(
                            //std::vector<Ptr<Vocab>> vocabs,
                            Ptr<Config> options,
                            Args ...args)
       : MNISTValidator(options),
         builder_(New<Builder>(options, keywords::inference=true, args...)) {
        initLastBest();
      }

      virtual float validateBG(Ptr<ExpressionGraph> graph,
                               Ptr<MNISTBatchGenerator<MNIST>> batchGenerator) {
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
        //auto model = options_->get<std::string>("model");
        //builder_->save(graph, model + ".best-" + type() + ".npz", true);
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
                                std::max_element(probs.begin() + i, probs.begin() + i + numLabels));
      if (pred == labels[i / numLabels])
        ++numCorrect;
    }
    return numCorrect;
  }
  };


}
