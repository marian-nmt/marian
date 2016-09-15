#pragma once

#include "node.h"
#include "tensor_operators.h"

namespace marian {

struct InputNode : public Node {
  template <typename ...Args>
  InputNode(Args ...args)
  : Node(args...) {
    UTIL_THROW_IF2(!Has(keywords::shape) &&
                   !Has(keywords::lazy_shape),
                   "Data items require shape information");
  }

  virtual void setVal(Tensor t)  {
    val_ = t;
    shape_ = t.shape();
    //@todo, shape checking
  };

  void forward() {}
  void backward() {}
};

struct ConstantNode : public Node {
  template <typename ...Args>
  ConstantNode(Args ...args)
  : Node(args...) {
    UTIL_THROW_IF2(!Has(keywords::shape) &&
                   !Has(keywords::lazy_shape),
                   "Constant items require shape information");
  }

  void forward() {}
  void backward() {}
};

struct ParamNode : public Node {
  template <typename ...Args>
  ParamNode(Args ...args)
  : Node(args...),
    init_(Get<std::function<void(Tensor)>>(keywords::init, [](Tensor){ })),
    initialized_(false)
  {
    UTIL_THROW_IF2(!Has(keywords::shape) &&
                   !Has(keywords::lazy_shape),
                   "Param items require shape information");
  }

  void forward() {}
  void backward() {}
  
  virtual void allocate(size_t batchSize) {
    val_.allocate(shape_);
    if(!initialized_) {
      init_(val_);
      initialized_ = true;
    }
  }

  private:
    std::function<void(Tensor)> init_;
    bool initialized_;
};

struct UnaryNodeOp : public Node {
    ChainPtr a_;

    template <typename ...Args>
    UnaryNodeOp(ChainPtr a, Args ...args)
    : Node(keywords::shape=a->shape(), //@TODO: Check keywords?
           args...),
    a_(a) {}
};

struct LogitNodeOp : public UnaryNodeOp {
  template <typename ...Args>
  LogitNodeOp(Args ...args)
  : UnaryNodeOp(args...) {  }

  void forward() {
    Element(_1 = Sigma(_2),
            val_, a_->val());
  }

  void backward() {
    Element(_1 += _2 * _3 * (1 - _3),
            a_->grad(), adj_, val_);
  }
};

struct TanhNodeOp : public UnaryNodeOp {
  template <typename ...Args>
  TanhNodeOp(Args ...args)
  : UnaryNodeOp(args...) { }

  void forward() {
    Element(_1 = Tanh(_2),
            val_, a_->val());
  }

  void backward() {
    Element(_1 += _2 * (1 - _3 * _3),
            a_->grad(), adj_, val_);
  }
};

// @TODO, make this numerically safe(r):
// softmax(X) = softmax_safe(X - max(X, axis=1))
// Probably best to do this directly in Softmax
// function. 
struct SoftmaxNodeOp : public UnaryNodeOp {
  template <typename ...Args>
    SoftmaxNodeOp(Args ...args)
    : UnaryNodeOp(args...) { }

  void forward() {
    // B = softmax(A).
    val_ = a_->val();
    Softmax(&val_);
  }

  void backward() {
    // For each row, the Jacobian times vector is given by:
    // J * dy = p .* (dy - avg*1)
    // where avg = p'*dy and p is the softmax output (probabilities).
    Tensor result = adj_;
    SubtractMean(&result, val_);
    // beta set to 1.0 in gemm, C = alpha * dot(A,B) + beta * C
    // to sum gradients from different graph parts.
    Prod(a_->grad(), adj_, result, false, false, 1.0);
  }
};

struct LogNodeOp : public UnaryNodeOp {
  template <typename ...Args>
  LogNodeOp(Args ...args)
  : UnaryNodeOp(args...) {}

  void forward() {
    Element(_1 = Log(_2), val_, a_->val());
  }

  void backward() {
    Element(_1 += _2 * 1.f / _3,
            a_->grad(), adj_, a_->val());
  }
};

struct ExpNodeOp : public UnaryNodeOp {
  template <typename ...Args>
    ExpNodeOp(Args ...args)
    : UnaryNodeOp(args...) { }

  void forward() {
    Element(_1 = Exp(_2), val_, a_->val());
  }

  void backward() {
    Element(_1 += _2 * Exp(_3),
            a_->grad(), adj_, a_->val());
  }
};

struct NegNodeOp : public UnaryNodeOp {
  template <typename ...Args>
  NegNodeOp(Args ...args)
  : UnaryNodeOp(args...) { }

  void forward() {
    Element(_1 = -_2, val_, a_->val());
  }

  void backward() {
    Element(_1 += -_2, a_->grad(), adj_);
  }
};

/******************************************************/

struct BinaryNodeOp : public Node {
  ChainPtr a_;
  ChainPtr b_;

  template <typename ...Args>
  BinaryNodeOp(ChainPtr a, ChainPtr b, Args ...args)
   : Node(args...), a_(a), b_(b) {}
};

/*** Matrix Product ***/

struct DotNodeOp : public BinaryNodeOp {
  template <typename ...Args>
  DotNodeOp(ChainPtr a, ChainPtr b, Args ...args)
  : BinaryNodeOp(
                 a, b,
                 keywords::shape=newShape(a, b),
                 args...) { }

  Shape newShape(ChainPtr a, ChainPtr b) {
    Shape shape1 = a->shape();
    Shape shape2 = b->shape();
    UTIL_THROW_IF2(shape1[1] != shape2[0],
                   "matrix product requires dimensions to match");
    shape1[1] = shape2[1];
    return shape1;
  }

  void forward() {
    // C = A*B
    Prod(val_, a_->val(), b_->val(), false, false);
  }

  void backward() {
    // D is the adjoint, the matrix of derivatives
    // df/dA += D*B.T
    // df/dB += A.T*D
    // beta set to 1.0 in gemm, C = alpha * dot(A,B) + beta * C
    // to sum gradients from different graph parts
    Prod(a_->grad(), adj_, b_->val(), false, true, 1.0);
    Prod(b_->grad(), a_->val(), adj_, true, false, 1.0);
  }
};

//struct BroadcastingNodeOp : public BinaryNodeOp {
//  template <typename ...Args>
//  BroadcastingNodeOp(ChainPtr a, ChainPtr b, Args ...args)
//  : BinaryNodeOp(broadcast(newShape(a ,b), a),
//                 broadcast(newShape(a ,b), b),
//                 keywords::shape=newShape(a, b),
//                 args...) {}
//  
//  static Shape newShape(ChainPtr a, ChainPtr b) {
//    size_t dimsA = a->shape().size();
//    size_t dimsB = b->shape().size();
//    UTIL_THROW_IF2(dimsA != dimsB,
//                   "Tensors have different numbers of dimensions");
//    Shape shape(dimsA);
//    for(size_t i = 0; i < dimsA; ++i) {
//      int dimA = a->shape()[i];
//      int dimB = b->shape()[i];
//      bool broadcastable = (dimA == dimB || dimA == 1 || dimB == 1);
//      UTIL_THROW_IF2(!broadcastable, "Different dimensions in elementwise "
//                     << "operation cannot be broadcasted: " << dimA << " != " << dimB);
//      shape[i] = std::max(dimA, dimB);
//      if(dimA == whatevs || dimB == whatevs)
//        shape[i] = whatevs;
//    }
//    return shape;
//  }
//};


struct PlusNodeOp : public BinaryNodeOp {
  template <typename ...Args>
  PlusNodeOp(ChainPtr a, ChainPtr b, Args ...args)
    : BinaryNodeOp(a, b, keywords::shape=a->shape(), args...) { }
    
  void forward() {
    Element(_1 = _2 + _3,
            val_, a_->val(), b_->val());
  }
  
  void backward() {
    Element(_1 += _2,
            a_->grad(), adj_);
    Element(_1 += _2,
            b_->grad(), adj_);
  }
};

struct MinusNodeOp : public BinaryNodeOp {
  template <typename ...Args>
  MinusNodeOp(ChainPtr a, ChainPtr b, Args ...args)
    : BinaryNodeOp(a, b, keywords::shape=a->shape(), args...) { }
    
  void forward() {
    Element(_1 = _2 - _3,
            val_, a_->val(), b_->val());
  }
  
  void backward() {
    Element(_1 += _2,
            a_->grad(), adj_);
    Element(_1 -= _2,
            b_->grad(), adj_);
  }
};

struct MultNodeOp : public BinaryNodeOp {
  template <typename ...Args>
  MultNodeOp(ChainPtr a, ChainPtr b, Args ...args)
    : BinaryNodeOp(a, b, keywords::shape=a->shape(), args...) { }
    
  void forward() {
    Element(_1 = _2 * _3,
            val_, a_->val(), b_->val());
  }
  
  void backward() {
    Element(_1 += _2 * _3,
            a_->grad(), adj_, b_->val());
    Element(_1 += _2 * _3,
            b_->grad(), adj_, a_->val());
  }
};

struct DivNodeOp : public BinaryNodeOp {
  template <typename ...Args>
  DivNodeOp(ChainPtr a, ChainPtr b, Args ...args)
    : BinaryNodeOp(a, b, keywords::shape=a->shape(), args...) { }
    
  void forward() {
    Element(_1 = _2 / _3,
            val_, a_->val(), b_->val());
  }
  
  void backward() {
    Element(_1 += _2 * 1.0f / _3,
            a_->grad(), adj_, b_->val());
    Element(_1 -= _2 * _3 / (_4 * _4),
            b_->grad(), adj_, a_->val(), b_->val());
  }  
};

}
