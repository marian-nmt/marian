#pragma once

#include <iostream>
#include <iomanip>
#include <boost/timer/timer.hpp>

#include "common/keywords.h"
#include "common/definitions.h"
#include "graph/expression_graph.h"
#include "optimizers/optimizers.h"

#include "examples/mnist/batch_generator.h"
#include "examples/mnist/feedforward.h"


using namespace data;
using namespace keywords;
using namespace models;

namespace marian {

class RunBase {
  public:
    virtual void run() = 0;

};

template <class DataSet>
class Trainer : public RunBase,
                public keywords::Keywords {

  public:
    template <typename ...Args>
    Trainer(Ptr<ExpressionGraph> graph,
            std::vector<int> dims,
            Ptr<DataSet> dataset,
            Args... args)
     : Keywords(args...),
       graph_(graph),
       dims_(dims),
       dataset_(dataset)
    {}

    void run() {
        boost::timer::cpu_timer trainTimer;

        auto opt = Get(optimizer, Optimizer<Adam>(0.002));
        auto validator = Get(valid, RunBasePtr());
        auto batchSize = Get(batch_size, 200);
        auto maxEpochs = Get(max_epochs, 20);
        MNISTBatchGenerator<DataSet> bg(dataset_, batchSize);

        size_t update = 0;
        for(int epoch = 1; epoch <= maxEpochs; ++epoch) {
          boost::timer::cpu_timer epochTimer;
          bg.prepare();

          float cost = 0;
          float totalExamples = 0;
          while(bg) {
            auto batch = bg.next();

            auto crossEntropy = FeedforwardClassifier(graph_, dims_ , batch, true);
            graph_->forward();
            graph_->backward();

            opt->update(graph_);

            cost += crossEntropy->scalar() * batch->dim();
            totalExamples += batch->dim();
            update++;
          }

          cost = cost / totalExamples;
          std::cerr << "Epoch: " << std::setw(std::to_string(maxEpochs).size())
            << epoch << "/" << maxEpochs << " - Update: " << update
            << " - Cost: " << std::fixed << std::setprecision(4) << cost
            << " - Time: " << epochTimer.format(2, "%ws")
            << " - " << trainTimer.format(0, "%ws") << std::endl;

          if(validator)
            validator->run();
        }
    }

  private:
    Ptr<ExpressionGraph> graph_;
    std::vector<int> dims_;
    Ptr<DataSet> dataset_;
};

template <class DataSet>
class Tester : public RunBase,
                  public keywords::Keywords {

  public:
    template <typename ...Args>
    Tester(Ptr<ExpressionGraph> graph,
           std::vector<int> dims,
           Ptr<DataSet> dataset,
           Args... args)
     : Keywords(args...),
       graph_(graph),
       dims_(dims),
       dataset_(dataset)
    {}

    void run() {
      auto batchSize = Get(batch_size, 200);
      MNISTBatchGenerator<DataSet> bg(dataset_, batchSize);

      bg.prepare(false);

      float total = 0;
      float cor = 0;
      while(bg) {
        auto batch = bg.next();

        auto ff = FeedforwardClassifier(graph_, dims_ , batch, false);
        graph_->forward();

        std::vector<float> scores;
        ff->val()->get(scores);
        cor += countCorrect(scores, batch->inputs()[1].data());
        total += batch->dim();
      }
      std::cerr << "Accuracy: " << cor / total << std::endl;
    }

  private:
    Ptr<ExpressionGraph> graph_;
    std::vector<int> dims_;
    Ptr<DataSet> dataset_;

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

template <class Process, typename ...Args>
RunBasePtr Run(Args&& ...args) {
  return RunBasePtr(new Process(args...));
}

}
