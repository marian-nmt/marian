#include <sstream>
#include "graph/backend_gpu.h"
#include "graph/expression_graph.h"

//#include "kernels/dropout.h"
//#include "kernels/tensor_operators.h"

namespace marian {

ExpressionGraph::ExpressionGraph(bool inference)
    : inferenceOnly_(inference), backend_(nullptr) {}

void ExpressionGraph::setDevice(DeviceId deviceId) {
  backend_ = New<BackendGPU>(deviceId, Config::seed);
  
  params_ = New<Parameters>();
  params_->init(backend_->getDevice());

  tensors_ = New<TensorAllocator>(backend_->getDevice());
}

Expr ExpressionGraph::dropout(float prob, Shape shape) {
  ABORT("Not implemented");
  
  //auto dropoutInit = [prob, this](Tensor t) {
  //  Dropout(t, prob, std::static_pointer_cast<BackendGPU>(backend_)->getCurandGenerator());
  //};
  //
  //return Expression<ConstantNode>(shared_from_this(),
  //                                keywords::init = dropoutInit,
  //                                keywords::shape = shape);
}

void ExpressionGraph::checkNan(Tensor t) {
  //ABORT_IF(throwNaN_ && IsNan(t), "Tensor has NaN");
}
}
