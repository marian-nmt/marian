#include "graph/expression_graph.h"
#include <sstream>

#include "tensors/tensor_operators.h"

namespace marian {

ExpressionGraph::ExpressionGraph(bool inference, bool optimized)
    : inferenceOnly_(inference), optimized_(optimized), backend_(nullptr) {}

void ExpressionGraph::setDevice(DeviceId deviceId) {
  if(!backend_) {
    backend_ = BackendByDevice(deviceId, Config::seed);
    params_ = New<Parameters>();
    params_->init(backend_);
    tensors_ = New<Tensors>(backend_);
  }
}

Expr ExpressionGraph::dropout(float prob, const Shape& shape) {
  return Expression<ConstantNode>(
      shared_from_this(), shape, [prob, this](Tensor t) { Dropout(t, prob); });
}

void ExpressionGraph::checkNan(Tensor t) {
  ABORT_IF(throwNaN_, "Not implemented");
  // ABORT_IF(throwNaN_ && IsNan(t), "Tensor has NaN");
}
}
