#include <stdio.h>
#include <stdlib.h>
#include <boost/chrono.hpp>
#include <boost/timer/timer.hpp>
#include <vector>

#include "marian.h"

using namespace marian;
using namespace keywords;

int main(int argc, char** argv) {
  auto c = New<Config>(argc, argv);

  auto g = New<ExpressionGraph>();
  g->setDevice({0, DeviceType::cpu});
  g->reserveWorkspaceMB(512);

  for(int i = 0; i < 10; ++i) {
    g->clear();
    auto mask = g->dropout(0.2, {10, 3072});
    debug(mask, "mask");
    g->forward();
  }

  return 0;
}
