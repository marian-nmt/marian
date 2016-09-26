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

template <typename ...Args>
void Train(ExpressionGraph& g,
          DataBasePtr dataset,
          const Args... args)
{
  using namespace keywords;
  using namespace data;
  Keywords keys(args...);
  OptimizerBasePtr opt = keys.Get(optimizer, Optimizer<Adam>());
  int batchSize = keys.Get(batch_size, 200);
  int maxEpochs = keys.Get(max_epochs, 50);
  //int batchSize = Get(batch_size, 200, args...);
  //int maxEpochs = Get(max_epochs, 50, args...);

  std::cerr << batchSize << std::endl;

  boost::timer::cpu_timer trainTimer;
  BatchGenerator bg(dataset, batchSize);
  size_t update = 0;
  for(int epoch = 1; epoch <= maxEpochs; ++epoch) {
    boost::timer::cpu_timer epochTimer;
    bg.prepare();

    float cost = 0;
    float totalExamples = 0;
    while(bg) {
      BatchPtr batch = bg.next();
      (*opt)(g, batch);
      cost += g["cost"].val()[0] * batch->dim();
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

}
