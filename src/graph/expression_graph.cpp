#include <sstream>
#include "graph/backend_gpu.h"
#include "graph/expression_graph.h"

#include "backend/dispatch.h"

namespace marian {

ExpressionGraph::ExpressionGraph(bool inference)
    : inferenceOnly_(inference), backend_(nullptr) {}

void ExpressionGraph::setDevice(DeviceId deviceId) {
  if(!backend_) {
    backend_ = New<BackendGPU>(deviceId, Config::seed);
  
    params_ = New<Parameters>();
    params_->init(backend_->getDevice());
  
    tensors_ = New<TensorAllocator>(backend_->getDevice());
  }
}

Expr ExpressionGraph::dropout(float prob, Shape shape) {
  return Expression<ConstantNode>(shared_from_this(),
                                  keywords::init = [prob, this](Tensor t) {
                                    Dropout(backend_, t, prob);
                                  },
                                  keywords::shape = shape);
}

void ExpressionGraph::checkNan(Tensor t) {
  ABORT_IF(throwNaN_, "Not implemented");
  //ABORT_IF(throwNaN_ && IsNan(t), "Tensor has NaN");
}
}
