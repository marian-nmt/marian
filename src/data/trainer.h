#pragma once

#include <iostream>
#include <iomanip>
#include <boost/timer/timer.hpp>

#include "common/keywords.h"
#include "common/definitions.h"
#include "graph/expression_graph.h"
#include "optimizers/optimizers.h"
#include "data/batch_generator.h"

namespace marian {

class RunBase {
  public:
    virtual void run() = 0;
};

typedef std::shared_ptr<RunBase> RunBasePtr;

template <class DataSet>
class Trainer : public RunBase,
                public keywords::Keywords {
  private:
    ExpressionGraphPtr graph_;
    std::shared_ptr<DataSet> dataset_;

  public:
    template <typename ...Args>
    Trainer(ExpressionGraphPtr graph,
            std::shared_ptr<DataSet> dataset,
            Args... args)
     : Keywords(args...),
       graph_(graph),
       dataset_(dataset)
    {}

    void run() {
        using namespace data;
        using namespace keywords;
        boost::timer::cpu_timer trainTimer;

        auto opt = Get(optimizer, Optimizer<Adam>());
        auto batchSize = Get(batch_size, 200);
        auto maxEpochs = Get(max_epochs, 50);
        BatchGenerator<DataSet> bg(dataset_, batchSize);

        auto validator = Get(valid, RunBasePtr());

        size_t update = 0;
        for(int epoch = 1; epoch <= maxEpochs; ++epoch) {
          boost::timer::cpu_timer epochTimer;
          bg.prepare();

          float cost = 0;
          float totalExamples = 0;
          while(bg) {
            auto batch = bg.next();
            opt->update(graph_);
            cost += graph_->get("cost")->val()->scalar() * batch->dim();
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
};

template <class DataSet>
class Validator : public RunBase,
                  public keywords::Keywords {
  private:
    ExpressionGraphPtr graph_;
    std::shared_ptr<DataSet> dataset_;

    float correct(const std::vector<float> pred, const std::vector<float> labels) {
      size_t num = labels.size();
      size_t scores = pred.size() / num;
      size_t acc = 0;
      for (size_t i = 0; i < num; ++i) {
        size_t proposed = 0;
        for(size_t j = 0; j < scores; ++j) {
          if(pred[i * scores + j] > pred[i * scores + proposed])
            proposed = j;
        }
        acc += (proposed == labels[i]);
      }
      return (float)acc;
    }

  public:
    template <typename ...Args>
    Validator(ExpressionGraphPtr graph,
              std::shared_ptr<DataSet> dataset,
              Args... args)
     : Keywords(args...),
       graph_(graph),
       dataset_(dataset)
    {}

    void run() {
        using namespace data;
        using namespace keywords;

        auto batchSize = Get(batch_size, 200);
        BatchGenerator<DataSet> bg(dataset_, batchSize);

        size_t update = 0;
        bg.prepare(false);

        float total = 0;
        float cor = 0;
        while(bg) {
            auto batch = bg.next();
            graph_->forward();
            std::vector<float> scores;
            graph_->get("scores")->val()->get(scores);

            cor += correct(scores, batch->inputs()[1].data());
            total += batch->dim();
            update++;
        }
        std::cerr << "Accuracy: " << cor / total << std::endl;
    }
};

template <class Process, typename ...Args>
RunBasePtr Run(Args&& ...args) {
  return RunBasePtr(new Process(args...));
}

}
