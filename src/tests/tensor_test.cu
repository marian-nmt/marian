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

  std::vector<Expr> x;
  for(int i = 0; i < 5; i++)
    x.push_back(graph->constant({1, 1}, keywords::init=inits::from_value(1)));
  auto input = concatenate(x, keywords::axis=2);

  auto mask = graph->constant({1, 1, 5}, keywords::init=inits::from_vector(std::vector<float>({1.f, 1.f, 1.f, 1.f, 0.f})));

  auto rnnFw = rnn::rnn(graph)
               ("type", "gru")
               ("direction", rnn::dir::alternating_forward)
               ("dimInput", 1)
               ("dimState", 1)
               ("skip", true)
               .push_back(rnn::cell(graph)("prefix", "l1"))
               .push_back(rnn::cell(graph)("prefix", "l2"))
               .push_back(rnn::cell(graph)("prefix", "l3"))
               .construct();

  auto output = rnnFw->transduce(input, mask);

  debug(input, "input");
  debug(mask, "mask");

  graph->forward();

  return 0;
}
