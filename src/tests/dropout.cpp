#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "marian.h"

using namespace marian;

int main(int argc, char** argv) {
  auto c = parseOptions(argc, argv, cli::mode::scoring, false);

  auto type = c->get<size_t>("cpu-threads") > 0
    ? DeviceType::cpu
    : DeviceType::gpu;
  DeviceId deviceId{0, type};

  auto g = New<ExpressionGraph>();
  g->setDevice(deviceId);
  g->reserveWorkspaceMB(512);

  for(int i = 0; i < 10; ++i) {
    g->clear();
    auto mask = g->dropoutMask(0.2, {1000, 16384});
    debug(mask, "mask");
    g->forward();
  }

  return 0;
}
