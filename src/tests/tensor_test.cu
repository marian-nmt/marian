#include <boost/timer/timer.hpp>
#include <iostream>
#include <map>

#include "marian.h"


int main(int argc, char** argv) {
  using namespace marian;
  using namespace keywords;

  Config::seed = 1;

  createLoggers(nullptr);

  auto g = New<ExpressionGraph>();
  g->setDevice(0);
  g->reserveWorkspaceMB(128);

  auto w = g->param("W", {10, 10}, init=inits::uniform());
  auto x = g->constant({100000, 10}, init=inits::uniform());

  std::vector<size_t> indices;
  while(indices.size() < 10)
    indices.push_back(rand() % 100000);

  auto xr = rows(x, indices);

  auto b = g->param("b", {1, 10}, init=inits::uniform());

  auto l1 = tanh(affine(xr, w, b));

  debug(w, "w");
  debug(b, "b");
  debug(x, "x");
  debug(l1, "l1");

  std::vector<size_t> labels({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});

  auto y = g->constant({10}, init = inits::from_vector(labels));
  auto cost = mean(cross_entropy(l1, y), axis = 0);

  debug(cost, "cost");

  g->forward();
  g->backward();

  return 0;
}
