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

  auto emb = graph->param("emb", {5, 5, 2}, keywords::init=inits::from_value(1));
  auto input = 2 * emb;

  auto ce = sum(reshape(input, {5*2, 5}), keywords::axis=1);


  debug(ce, "cost");
  debug(input, "input");
  debug(emb, "emb");

  graph->forward();
  graph->backward();

  return 0;
}
