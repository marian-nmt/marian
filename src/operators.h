#pragma once

#include <memory>
#include <functional>
#include <vector>
#include <cmath>

#include "marian.h"
#include "cudnn_tensor.h"

namespace marian {

/*** Unary operators ***/

struct UnaryNodeOp : public Node {
    ChainPtr a_;
    
    UnaryNodeOp(const Tensor t, ChainPtr a)
    : Node(t), a_(a) {}
};

struct SigmaNodeOp : public UnaryNodeOp {
  SigmaNodeOp(ChainPtr a)
  : UnaryNodeOp(Tensor(a->val().shape()), a) { }
  
  void forward() {
    Element(_1 = Sigma(_2),
            val_, a_->val());
  }
  
  void backward() {
    Element(_1 += _2 * Sigma(_3) * (1 - Sigma(_3)),
            a_->grad(), adj_, a_->val());
  }
};

inline Var sigma(Var a) {
  return Var(new SigmaNodeOp(a));
}

struct TanhNodeOp : public UnaryNodeOp {
  TanhNodeOp(ChainPtr a)
  : UnaryNodeOp(Tensor(a->val().shape()), a) { }
  
  void forward() {
    Element(_1 = Tanh(_2),
            val_, a_->val());
  }
  
  void backward() {
    Element(_1 += _2 * (1 - Tanh(_3) * Tanh(_3)),
            a_->grad(), adj_, a_->val());
  }
};

inline Var tanh(Var a) {
  return Var(new TanhNodeOp(a));
}

struct LogNodeOp : public UnaryNodeOp {
  LogNodeOp(ChainPtr a)
  : UnaryNodeOp(Tensor(a->val().shape()), a) { }
  
  void forward() {
    Element(_1 = Log(_2), val_, a_->val());
  }
  
  void backward() {
    Element(_1 += _2 * 1.f / _3,
            a_->grad(), adj_, a_->val());
  }
};

inline Var log(Var a) {
  return Var(new LogNodeOp(a));
};

struct ExpNodeOp : public UnaryNodeOp {
  ExpNodeOp(ChainPtr a)
  : UnaryNodeOp(Tensor(a->val().shape()), a) { }
  
  void forward() {
    Element(_1 = Exp(_2), val_, a_->val());
  }
  
  void backward() {
    Element(_1 += _2 * Exp(_3),
            a_->grad(), adj_, a_->val());
  }
};

inline Var exp(Var a) {
  return Var(new ExpNodeOp(a));
};

struct NegNodeOp : public UnaryNodeOp {
  NegNodeOp(ChainPtr a)
  : UnaryNodeOp(Tensor(a->val().shape()), a) { }
  
  void forward() {
    Element(_1 = -_2, val_, a_->val());
  }
  
  void backward() {
    Element(_1 += -_2, a_->grad(), adj_);
  }
};

inline Var operator-(Var a) {
  return Var(new NegNodeOp(a));
};

/******************************************************/

struct BinaryNodeOp : public Node {
  ChainPtr a_;
  ChainPtr b_;

  BinaryNodeOp(const Tensor t, ChainPtr a, ChainPtr b)
  : Node(t), a_(a), b_(b) {}
};

/*** Matrix Product ***/

struct DotNodeOp : public BinaryNodeOp {
  DotNodeOp(ChainPtr a, ChainPtr b) : BinaryNodeOp(Tensor(shape(a, b)), a, b) { }
  
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

inline Var dot(Var a, Var b) {
  return Var(new DotNodeOp(a, b));
}

/******************************************************/

Var broadcast(Shape shape, Var a) {
  if(a.val().shape() == shape) {
    return a;
  }
  else {
    size_t dimsA = a.val().shape().size();
    size_t dimsB = shape.size();
    UTIL_THROW_IF2(dimsA != dimsB,
                   "Tensor and shape have different number of dimensions");
    for(size_t i = 0; i < dimsA; ++i) {
      int dimA = a.val().shape()[i];
      int dimB = shape[i];
      bool broadcastable = (dimA == dimB || dimA == 1);
      UTIL_THROW_IF2(!broadcastable,
                     "Cannot broadcast tensor dimension "
                     << dimA << " to " << dimB);
      if(dimA == 1 && dimB > 1) {
        std::cerr << "Broadcasting dim " << i << " from " << dimA << " to " << dimB << std::endl;
        if(i == 0) {
          Var one = Tensor({shape[0], 1}, 1);
          a = dot(one, a);
        }
        else if(i == 1) {
          Var one = Tensor({1, shape[1]}, 1);
          a = dot(a, one);
        }
        else {
          UTIL_THROW2("Not implemented");        
        }
      }
    }
    return a;
  }
}

struct BroadcastingNodeOp : public BinaryNodeOp {
  BroadcastingNodeOp(Var a, Var b)
  : BroadcastingNodeOp(Tensor(shape(a ,b)), broadcast(shape(a ,b), a), broadcast(shape(a ,b), b)) {}
  
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
  
  private:
    BroadcastingNodeOp(const Tensor t, ChainPtr a, ChainPtr b)
    : BinaryNodeOp(t, a, b) {}  
};

/*** Binary arithmetic ***/

/*** Plus ***/

struct PlusNodeOp : public BroadcastingNodeOp {
  PlusNodeOp(Var a, Var b) : BroadcastingNodeOp(a, b) { }
  
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

inline Var operator+(Var a, Var b) {
  return Var(new PlusNodeOp(a, b));
}

/*** Minus ***/

struct MinusNodeOp : public BroadcastingNodeOp {
  MinusNodeOp(Var a, Var b) : BroadcastingNodeOp(a, b) { }
  
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

inline Var operator-(Var a, Var b) {
  return Var(new MinusNodeOp(a, b));
}

/*** Mult ***/

struct MultNodeOp : public BroadcastingNodeOp {
  MultNodeOp(Var a, Var b) : BroadcastingNodeOp(a, b) { }
  
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

inline Var operator*(Var a, Var b) {
  return Var(new MultNodeOp(a, b));
}

/*** Division ***/

struct DivNodeOp : public BroadcastingNodeOp {
  DivNodeOp(Var a, Var b) : BroadcastingNodeOp(a, b) { }  
  
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

inline Var operator/(Var a, Var b) {
  return Var(new DivNodeOp(a, b));
}


/*** Reductions ***/

enum Axis { undef, axis0, axis1, axis2, axis3 };

// inefficient
inline Var sum(Var a, Axis axis = Axis::undef) {
  if(axis == Axis::axis0) {
    int rows = a.val().shape()[0];
    int cols = a.val().shape()[1]; 
    Var one = Tensor({1, rows}, 1);
    return dot(one, a);        
  }
  else if(axis == Axis::axis1) {
    int rows = a.val().shape()[0];
    int cols = a.val().shape()[1]; 
    Var one = Tensor({cols, 1}, 1);
    return dot(a, one);          
  }
  else if(axis == Axis::axis2) {
    UTIL_THROW2("Not implemented");
  }
  else if(axis == Axis::axis3) {
    UTIL_THROW2("Not implemented");
  }
  return sum(sum(a, Axis::axis0), Axis::axis1);
}

// inefficient
inline Var softmax(Var a, Axis axis = Axis::undef) {
    Var e = exp(a);
    return e / sum(e, axis);
}

// inefficient
inline Var mean(Var a, Axis axis = Axis::undef) {
    switch (axis) {
      case Axis::axis0:
        return sum(a, axis) / a.val().shape()[0];
      case Axis::axis1:
        return sum(a, axis) / a.val().shape()[1];
      case Axis::axis2:
        UTIL_THROW2("Not implemented");
      case Axis::axis3:
        UTIL_THROW2("Not implemented");
      case Axis::undef:
      default:
        return sum(a) / a.val().size();
    }
}

}