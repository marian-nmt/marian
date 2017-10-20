#include <boost/timer/timer.hpp>
#include <iostream>
#include <map>

#include "marian.h"
#include "rnn/rnn.h"

int main(int argc, char** argv) {
  using namespace marian;
  using namespace keywords;

  auto graph = New<ExpressionGraph>();
  graph->setDevice(0);

  auto in1 = graph->constant({2, 3, 2, 1}, init=inits::from_value(1));
  auto in2 = graph->constant({2, 3, 2, 1}, init=inits::from_value(2));
  auto in3 = graph->constant({2, 3, 2, 1}, init=inits::from_value(3));
  auto in4 = graph->constant({2, 3, 2, 1}, init=inits::from_value(4));

  auto out = concatenate2({in1, in2, in3, in4}, axis=1);

  debug(in1, "in1");
  debug(in2, "in2");
  debug(in3, "in3");
  debug(in4, "in4");
  debug(out, "out");

  graph->forward();

  return 0;
}
