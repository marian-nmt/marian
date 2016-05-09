#pragma once

#include "graph.h"
#include "expressions.h"
//#include "expression_operators.h"

namespace marian {

struct DataNode : public Node {
  template <typename ...Args>
  DataNode(Args ...args)
  : Node(args...) { }
  
  void forward() {}
  void backward() {}
};

struct ConstantNode : public Node {
  template <typename ...Args>
  ConstantNode(Args ...args)
  : Node(args...) { }
  
  void forward() {}
  void backward() {}
};

struct ParamNode : public Node {
  template <typename ...Args>
  ParamNode(Args ...args)
  : Node(args...),
    init_(Get<std::function<void(Tensor)>>(keywords::init, [](Tensor){ }))
  { }
  
  void forward() {}
  void backward() {}
  
  virtual void allocate() {
    val_.allocate(shape_);
    init_(val_);
  }
  
  private:
    std::function<void(Tensor)> init_;
};

struct UnaryNodeOp : public Node {
    ChainPtr a_;
    
    template <typename ...Args>
    UnaryNodeOp(ChainPtr a, Args ...args)
    : Node(args...), a_(a) {}
};

struct SigmoidNodeOp : public UnaryNodeOp {
  template <typename ...Args>
  SigmoidNodeOp(Args ...args)
  : UnaryNodeOp(args...) {  }
  
  void forward() {
    Element(_1 = Sigma(_2),
            val_, a_->val());
  }
  
  void backward() {
    Element(_1 += _2 * Sigma(_3) * (1 - Sigma(_3)),
            a_->grad(), adj_, a_->val());
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
    Element(_1 += _2 * (1 - Tanh(_3) * Tanh(_3)),
            a_->grad(), adj_, a_->val());
  }
};

struct LogNodeOp : public UnaryNodeOp {
  template <typename ...Args>
  LogNodeOp(Args ...args)
  : UnaryNodeOp(args...) {
    std::cerr << "log" << std::endl;
  }
  
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
  : BinaryNodeOp(a, b, args...) { }
  
  Shape shape(ChainPtr a, ChainPtr b) {
    UTIL_THROW_IF2(a->val().shape()[1] != b->val().shape()[0],
                   "matrix product requires dimensions to match");
    Shape shape1 = a->val().shape();
    Shape shape2 = b->val().shape();
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

Expr broadcast(Shape shape, Expr a);

struct BroadcastingNodeOp : public BinaryNodeOp {
  template <typename ...Args>
  BroadcastingNodeOp(Expr a, Expr b, Args ...args)
  : BinaryNodeOp(broadcast(shape(a ,b), a),
                       broadcast(shape(a ,b), b),
                       args...) {}
  
  static Shape shape(ChainPtr a, ChainPtr b) {
    size_t dimsA = a->val().shape().size();
    size_t dimsB = b->val().shape().size();
    UTIL_THROW_IF2(dimsA != dimsB,
                   "Tensors have different numbers of dimensions");
    Shape shape(dimsA);
    for(size_t i = 0; i < dimsA; ++i) {
      int dimA = a->val().shape()[i];
      int dimB = b->val().shape()[i];
      bool broadcastable = (dimA == dimB || dimA == 1 || dimB == 1);
      UTIL_THROW_IF2(!broadcastable, "Different dimensions in elementwise "
                     << "operation cannot be broadcasted: " << dimA << " != " << dimB);
      shape[i] = std::max(dimA, dimB);
    }
    return shape;
  }
};


struct PlusNodeOp : public BroadcastingNodeOp {
  template <typename ...Args>
  PlusNodeOp(Args ...args) : BroadcastingNodeOp(args...) { }
  
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

struct MinusNodeOp : public BroadcastingNodeOp {
  template <typename ...Args>
  MinusNodeOp(Args ...args) : BroadcastingNodeOp(args...) { }
  
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

struct MultNodeOp : public BroadcastingNodeOp {
  template <typename ...Args>
  MultNodeOp(Args ...args) : BroadcastingNodeOp(args...) { }
  
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

struct DivNodeOp : public BroadcastingNodeOp {
  template <typename ...Args>
  DivNodeOp(Args ...args) : BroadcastingNodeOp(args...) { }
    
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