#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "marian.h"

using namespace marian;

int main(int argc, char** argv) {
  auto c = New<Config>(argc, argv);

  auto type = c->get<size_t>("cpu-threads") > 0
    ? DeviceType::cpu
    : DeviceType::gpu;
  DeviceId deviceId{0, type};

  auto g = New<ExpressionGraph>();
  g->setDevice(deviceId);
  g->reserveWorkspaceMB(512);

  for(int i = 0; i < 10; ++i) {
    g->clear();
    auto mask1 = g->dropoutMask(0.2, {10, 3072});
    auto mask2 = g->dropoutMask(0.3, {1, 3072});
    auto mask = mask1 + mask2;
    debug(mask1, "mask1");
    debug(mask2, "mask2");
    debug(mask, "mask");
    g->forward();
  }

  return 0;
}
