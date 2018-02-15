/* All or part of this file was contributed by Intel under license:
 *   Copyright (C) 2017-2018 Intel Corporation
 *   SPDX-License-Identifier: MIT
 */

#include "kernels/dropout.h"

#if MKL_FOUND
namespace marian {

namespace cpu {

void Dropout(Tensor tensor, float p, VSLStreamStatePtr gen) {
  int n = tensor->size();
  // FIXME: vsRngUniform is in [0, 1) vs. curandGeneratorUniform's (0, 1]
  vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, gen, n, tensor->data(), 0.f, 1.f);

  float not_p = 1.f - p;
  float* data = tensor->data();
  for (int i = 0; i< n; ++i) {
    data[i] = (data[i] < not_p) / not_p;
  }
}

void Gaussian(Tensor tensor, float mean, float stddev, VSLStreamStatePtr gen) {
  int n = tensor->size();
  vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2, gen, n, tensor->data(), mean, stddev);
}

}

}
#endif

#if MKL_FOUND || CUDA_FOUND
namespace marian {

void Dropout(Tensor tensor, float h, RNG gen) {
  #if MKL_FOUND
  if (tensor->residency == DEVICE_CPU) {
    cpu::Dropout(tensor, h, boost::get<VSLStreamStatePtr>(gen));
  }
  #endif

  #if CUDA_FOUND
  if (tensor->residency == DEVICE_GPU) {
    gpu::Dropout(tensor, h, boost::get<curandGenerator_t>(gen));
  }
  #endif
}

void Gaussian(Tensor tensor, float mean, float stddev, RNG gen) {
  #if MKL_FOUND
  if (tensor->residency == DEVICE_CPU) {
    cpu::Gaussian(tensor, mean, stddev, boost::get<VSLStreamStatePtr>(gen));
  }
  #endif

  #if CUDA_FOUND
  if (tensor->residency == DEVICE_GPU) {
    gpu::Gaussian(tensor, mean, stddev, boost::get<curandGenerator_t>(gen));
  }
  #endif
}

}
#endif
