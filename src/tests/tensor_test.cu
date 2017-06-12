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
  auto x = g->constant({10, 10}, init=inits::from_value(1));

  auto l1 = dot(x, w);

  debug(w, "w");
  debug(x, "x");
  debug(l1, "l1");

  g->forward();

  return 0;
}
