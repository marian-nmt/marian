#include "graph/expression_graph.h"
#include <sstream>

#include "tensors/tensor_operators.h"

namespace marian {

ExpressionGraph::ExpressionGraph(bool inference)
    : inferenceOnly_(inference), backend_(nullptr) {}

void ExpressionGraph::setDevice(DeviceId deviceId, Ptr<Device> device) {
  if(!backend_) {
    backend_ = BackendByDeviceId(deviceId, Config::seed);
    params_ = New<Parameters>();
    params_->init(backend_);
    if(device)
      tensors_ = New<Tensors>(backend_, device);
    else
      tensors_ = New<Tensors>(backend_);
  }
}

Expr ExpressionGraph::dropoutMask(float prob, const Shape& shape) {
  return constant(shape, inits::dropout(prob));
}

void ExpressionGraph::checkNan(Tensor t) {
  ABORT_IF(throwNaN_, "Not implemented"); t;
  // ABORT_IF(throwNaN_ && IsNan(t), "Tensor has NaN");
}

void ExpressionGraph::save(std::vector<io::Item>& ioItems) {
  for(auto p : params()->getMap()) {
    std::string pName = p.first;

    if(!namespace_.empty()) {
      if(pName.substr(0, namespace_.size() + 2) == namespace_ + "::")
        pName = pName.substr(namespace_.size() + 2);
    }

    ABORT_IF(p.second->val()->type() != Type::float32,
             "Only float32 supported at the moment");

    Tensor val = p.second->val();
    io::Item item;
    val->get(item, pName);
    ioItems.emplace_back(std::move(item));
  }
}

}  // namespace marian
