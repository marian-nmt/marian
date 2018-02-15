/* All or part of this file was contributed by Intel under license:
 *   Copyright (C) 2017-2018 Intel Corporation
 *   SPDX-License-Identifier: MIT
 */

#include <sstream>
#include "graph/expression_graph.h"
#include "kernels/dropout.h"

namespace marian {

ExpressionGraph::ExpressionGraph(ResidentDevice residency, bool inference)
    : inferenceOnly_(inference),
      #if CUDA_FOUND
      backend_(residency == DEVICE_CPU ? New<BackendCPU>() : New<BackendGPU>()),
      #else
      backend_(New<BackendCPU>()),
      #endif
      residency(residency) {}

void ExpressionGraph::setDevice(size_t device) {
  device_ = device;

  params_ = New<Parameters>();
  if (residency == DEVICE_CPU) {
    params_->init<TensorAllocatorCPU>(device_);
    tensors_.reset(new TensorAllocatorCPU(device));
  }
  #if CUDA_FOUND
  else {
    params_->init<TensorAllocatorGPU>(device_);
    tensors_.reset(new TensorAllocatorGPU(device));
  }
  #endif

  backend_->setHandles(device, Config::seed);
}

Expr ExpressionGraph::dropout(float prob, Shape shape) {
  auto dropoutInit = [prob, this](Tensor t) {
    Dropout(t, prob, backend_->getRNG());
  };

  return Expression<ConstantNode>(shared_from_this(),
                                  keywords::init = dropoutInit,
                                  keywords::shape = shape);
}

Expr ExpressionGraph::gaussian(float mean, float stddev, Shape shape) {
  auto gaussianInit = [mean, stddev, this](Tensor t) {
    Gaussian(t, mean, stddev, backend_->getRNG());
  };

  return Expression<ConstantNode>(shared_from_this(),
                                  keywords::init = gaussianInit,
                                  keywords::shape = shape);
}

}
