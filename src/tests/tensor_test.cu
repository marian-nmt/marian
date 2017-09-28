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

  std::vector<float> vals1 = {
    -87.41486359, -87.41487122, -87.41486359, -87.41486359, -87.41487885,
    -87.41485596, -87.41486359, -87.41485596, -87.41487122, -87.41486359,
    -87.41487885, -87.41486359, -87.41487885, -87.41487885, -87.41486359,
    -87.41486359
  };

  std::vector<float> vals;
  for(int i = 0; i < 16 * 16; ++i)
    for(auto v: vals1)
      vals.push_back(v);

  auto in = graph->constant({16, 16, 16}, init=inits::from_vector(vals));

  std::vector<float> vMask1(32, 1.f);
  vMask1[15] = 0.f;

  std::vector<float> vMask;
  for(int i = 0; i < 8; ++i)
    for(auto v: vMask1)
      vMask.push_back(v);

  auto inMask = graph->constant({16, 16, 1}, init=inits::from_vector(vMask));

  auto out = softmax(in, inMask);

  debug(in, "cost");
  debug(inMask, "mask");
  debug(out, "out");

  graph->forward();

  return 0;
}
