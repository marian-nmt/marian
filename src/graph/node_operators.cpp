#include "node_operators.h"
#include "expression_graph.h"

#include "tensors/tensor_operators.h"
#include "tensors/cpu/sharp/sse_gemm.h"

namespace marian {

size_t ConstantNode::allocate() {
  // @TODO params
  size_t elements = 0;
  if(!val_) {
    graph()->tensor(val_, shape_);
    elements = val_->shape().elements();
  }
  return elements;
}

void ConstantNode::init() {
  if(!initialized_) {
    (*init_)(val_);
    initialized_ = true;
  }
  init_.reset();
}

size_t ParamNode::allocate() {
  // @TODO params
  size_t elements = 0;
  if(!val_) {
    graph()->tensor(val_, shape_);
    elements = val_->shape().elements();
  }
  return elements;
}

void ParamNode::transposeAndQuantize() {
  Tensor temp;
  graph()->tensor(temp, shape_, Type::float32);
  (*init_)(temp);

  if(transpose_) {
    Tensor temp2;
    graph()->tensor(temp2, Shape{shape_[-1], shape_[-2]}, Type::float32);
    TransposeND(temp2, temp, {1, 0});
    graph()->free(temp);
    temp = temp2;
  }

  int num_rows = temp->shape()[-2];
  int width = temp->shape()[-1];
  double quant_mult = pow(2.0, 10.0);
  assert(width % 8 == 0);

  Quantize(temp->data(),
           val_->data<__m128i>(),
           (float)quant_mult,
           num_rows,
           width);

  graph()->free(temp);
}

void ParamNode::init() {
  if(!initialized_) {
    if(quantize_) {
      transposeAndQuantize();
    }
    else {
      (*init_)(val_);
    }
    initialized_ = true;
  }
  init_.reset();
}
}
