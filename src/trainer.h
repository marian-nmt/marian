#pragma once

#include <iostream>
#include <iomanip>
#include <boost/timer/timer.hpp>

#include "keywords.h"
#include "definitions.h"
#include "expression_graph.h"
#include "batch_generator.h"
#include "optimizers.h"

namespace marian {

class RunBase {
  public:
    virtual void run() = 0;
};

typedef std::shared_ptr<RunBase> RunBasePtr;

class Trainer : public RunBase,
                public keywords::Keywords {
  private:
    ExpressionGraphPtr graph_;
    data::DataBasePtr dataset_;

  public:
    template <typename ...Args>
    Trainer(ExpressionGraphPtr graph,
            data::DataBasePtr dataset,
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
        BatchGenerator bg(dataset_, batchSize);

        auto validator = Get(valid, RunBasePtr());

        size_t update = 0;
        for(int epoch = 1; epoch <= maxEpochs; ++epoch) {
          boost::timer::cpu_timer epochTimer;
          bg.prepare();

          float cost = 0;
          float totalExamples = 0;
          while(bg) {
            BatchPtr batch = bg.next();
            opt->update(graph_, batch);
            cost += (*graph_)["cost"].val()[0] * batch->dim();
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

class Validator : public RunBase,
                public keywords::Keywords {
  private:
    ExpressionGraphPtr graph_;
    data::DataBasePtr dataset_;

    float correct(const std::vector<float> pred, const std::vector<float> labels) {
      size_t acc = 0;
      for (size_t i = 0; i < labels.size(); i += 10) {
        size_t correct = 0;
        size_t proposed = 0;
        for (size_t j = 0; j < 10; ++j) {
          if (labels[i + j])
            correct = j;
          if (pred[i + j] > pred[i + proposed])
            proposed = j;
        }
        acc += (correct == proposed);
      }
      return (float)acc;
    }

  public:
    template <typename ...Args>
    Validator(ExpressionGraphPtr graph,
              data::DataBasePtr dataset,
              Args... args)
     : Keywords(args...),
       graph_(graph),
       dataset_(dataset)
    {}

    void run() {
        using namespace data;
        using namespace keywords;

        auto batchSize = Get(batch_size, 200);
        BatchGenerator bg(dataset_, batchSize);

        size_t update = 0;
        bg.prepare();

        float total = 0;
        float cor = 0;
        while(bg) {
            BatchPtr batch = bg.next();
            graph_->inference(batch);
            std::vector<float> scores;
            scores << (*graph_)["scores"].val();

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
