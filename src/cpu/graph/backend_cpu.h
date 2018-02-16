/* All or part of this file was contributed by Intel under license:
 *   Copyright (C) 2017-2018 Intel Corporation
 *   SPDX-License-Identifier: MIT
 */

#pragma once

#include "graph/backend.h"

#if MKL_FOUND
#include <mkl.h>
#endif

namespace marian {

class BackendCPU : public Backend {
  #if MKL_FOUND
  VSLStreamStatePtr gen;
  #endif

  public:
  BackendCPU() : Backend(DEVICE_CPU) {}

  void setDevice(size_t device) {}

  void setHandles(size_t device, size_t seed) {
    vslNewStream(&gen, VSL_BRNG_MT19937, seed);
  }

  #if MKL_FOUND
  RNG getRNG() {
    return gen;
  }
  #endif
};

}
