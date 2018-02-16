/* All or part of this file was contributed by Intel under license:
 *   Copyright (C) 2017-2018 Intel Corporation
 *   SPDX-License-Identifier: MIT
 */

#pragma once

#include "tensors/residency.h"
#include "kernels/dropout.h"

namespace marian {

struct Backend {
  const ResidentDevice residency;
  Backend(ResidentDevice residency) : residency(residency) {}

  virtual void setDevice(size_t device) = 0;

  virtual void setHandles(size_t device, size_t seed) = 0;

  #if MKL_FOUND || CUDA_FOUND
  virtual RNG getRNG() = 0;
  #endif
};

}

#include "graph/backend_cpu.h"

#if CUDA_FOUND
#include "graph/backend_gpu.h"
#endif
