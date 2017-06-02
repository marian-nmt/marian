#include <sstream>
#include "expression_graph.h"
#include "backend_gpu.h"
#include "kernels/dropout.h"

namespace marian {

    ExpressionGraph::ExpressionGraph(bool inference)
     : inferenceOnly_(inference),
       backend_(New<BackendGPU>()) {}

    void ExpressionGraph::setDevice(size_t device) {
      device_ = device;

      params_ = New<Parameters>();
      params_->init(device_);

      tensors_ = New<TensorAllocator>(device);
      
      std::static_pointer_cast<BackendGPU>(backend_)->setHandles(device, Config::seed);
    }
    
    Expr ExpressionGraph::dropout(float prob, Shape shape) {
      auto dropoutInit = [prob, this](Tensor t) {
        Dropout(t, prob, std::static_pointer_cast<BackendGPU>(backend_)->getCurandGenerator());
      };

      return Expression<ConstantNode>(shared_from_this(),
                                      keywords::init=dropoutInit,
                                      keywords::shape=shape);
    }
}
