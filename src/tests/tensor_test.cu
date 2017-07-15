#include <boost/timer/timer.hpp>
#include <iostream>
#include <map>

#include "marian.h"
#include "rnn/rnn.h"

int main(int argc, char** argv) {
  using namespace marian;

  marian::Config::seed = 1234;

  //marian::Config config(argc, argv);

  auto graph = New<ExpressionGraph>();
  graph->setDevice(0);

  auto output = graph->param("bla", {1, 5, 5}, keywords::init=inits::from_value(1));

  auto picks = graph->constant({1 * 5}, keywords::init=inits::from_value(1));
  auto cost = cross_entropy(output, picks);

  debug(cost, "cost");
  debug(output, "output");


  graph->forward();
  graph->backward();

  return 0;
}
