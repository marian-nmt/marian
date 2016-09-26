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
    ExpressionGraph& graph_;
    data::DataBasePtr dataset_;

  public:
    template <typename ...Args>
    Trainer(ExpressionGraph& graph,
            data::DataBasePtr dataset,
            Args... args)
     : Keywords(args...),
       graph_(graph),
       dataset_(dataset_)
    {}

    void run() {
        using namespace data;
        using namespace keywords;
        boost::timer::cpu_timer trainTimer;

        auto opt = Get(optimizer, Optimizer<Adam>());
        auto batchSize = Get(batch_size, 200);
        auto maxEpochs = Get(max_epochs, 50);
        BatchGenerator bg(dataset_, batchSize);

        size_t update = 0;
        for(int epoch = 1; epoch <= maxEpochs; ++epoch) {
          boost::timer::cpu_timer epochTimer;
          bg.prepare();

          float cost = 0;
          float totalExamples = 0;
          while(bg) {
            BatchPtr batch = bg.next();
            (*opt)(graph_, batch);
            cost += graph_["cost"].val()[0] * batch->dim();
            totalExamples += batch->dim();
            update++;
          }
          cost = cost / totalExamples;

          std::cerr << "Epoch: " << std::setw(std::to_string(maxEpochs).size())
            << epoch << "/" << maxEpochs << " - Update: " << update
            << " - Cost: " << std::fixed << std::setprecision(4) << cost
            << " - Time: " << epochTimer.format(2, "%ws")
            << " - " << trainTimer.format(0, "%ws") << std::endl;
        }
    }
};

template <class Process, typename ...Args>
RunBasePtr Run(Args ...args) {
  return RunBasePtr(new Process(args...));
}

}
