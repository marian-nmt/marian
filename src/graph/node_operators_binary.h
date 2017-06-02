#pragma once

#include <thread>

#include "graph/node.h"
#include "kernels/thrust_functions.h"
#include "kernels/tensor_operators.h"
#include "graph/backend_gpu.h"

namespace marian {

struct DotNodeOp : public NaryNodeOp {
  template <typename ...Args>
  DotNodeOp(Expr a, Expr b, Args ...args)
  : NaryNodeOp({a, b},
               keywords::shape=newShape(a, b),
               args...) { }

  Shape newShape(Expr a, Expr b) {

    auto shapeA = a->shape();
    auto shapeB = b->shape();

    Shape outShape = shapeA;
    outShape.set(1, shapeB[1]);
    UTIL_THROW_IF2(shapeA[1] != shapeB[0],
                 "matrix product requires dimensions to match");
    return outShape;
  }

  NodeOps forwardOps() {
    // C = A*B
    return {
      NodeOp(Prod(std::static_pointer_cast<BackendGPU>(getBackend())->getCublasHandle(),
                  val_,
                  child(0)->val(),
                  child(1)->val(),
                  false, false))
    };
  }

  NodeOps backwardOps() {
    // D is the adjoint, the matrix of derivatives
    // df/dA += D*B.T
    // df/dB += A.T*D
    // beta set to 1.0 in gemm, C = dot(A,B) + beta * C
    // to sum gradients from different graph parts
    return {
      NodeOp(Prod(std::static_pointer_cast<BackendGPU>(getBackend())->getCublasHandle(),
                  child(0)->grad(),
                  adj_,
                  child(1)->val(),
                  false, true, 1.0)),
      NodeOp(Prod(std::static_pointer_cast<BackendGPU>(getBackend())->getCublasHandle(),
                  child(1)->grad(),
                  child(0)->val(),
                  adj_,
                  true, false, 1.0))
    };
  }

  const std::string type() {
    return "•";
  }

  const std::string color() {
    return "orange";
  }
};

struct ScalarProductNodeOp : public NaryNodeOp {
  template <typename ...Args>
  ScalarProductNodeOp(Expr a, Expr b, Args ...args)
  : NaryNodeOp({a, b},
               keywords::shape=newShape(a, b, args...),
               args...) { }

  template <typename ...Args>
  Shape newShape(Expr a, Expr b, Args ...args) {
    int ax = keywords::Get(keywords::axis, -1, args...);
    Shape full = a->shape();
    for(int i = 0; i < b->shape().size(); ++i)
      full.set(i, std::max(full[i], b->shape()[i]));

    if(ax != -1) {
      full.set(ax, 1);
    }
    else {
      full.set(0, 1);
      full.set(1, 1);
      full.set(2, 1);
      full.set(3, 1);
    }
    return full;
  }

  NodeOps forwardOps() {
    return {
      NodeOp(Reduce(_1 * _2,
                    val_,
                    child(0)->val(),
                    child(1)->val()))
    };
  }

  NodeOps backwardOps() {
    return {
      NodeOp(Add(_1 * _2,
             child(0)->grad(),
             child(1)->val(),
             adj_)),
      NodeOp(Add(_1 * _2,
             child(1)->grad(),
             child(0)->val(),
             adj_))
    };
  }

  const std::string type() {
    return "scalar-product";
  }

  const std::string color() {
    return "orange";
  }

};


struct ElementBinaryNodeOp : public NaryNodeOp {
  template <typename ...Args>
  ElementBinaryNodeOp(Expr a, Expr b, Args ...args)
   : NaryNodeOp({a, b},
                keywords::shape=newShape(a, b),
                args...) {}

  Shape newShape(Expr a, Expr b) {
    Shape shape1 = a->shape();
    Shape shape2 = b->shape();
    for(int i = 0; i < shape1.size(); ++i) {
      UTIL_THROW_IF2(shape1[i] != shape2[i] && shape1[i] != 1 && shape2[i] != 1,
                     "Shapes cannot be broadcasted");
      shape1.set(i, std::max(shape1[i], shape2[i]));
    }
    return shape1;
  }

  const std::string color() {
    return "yellow";
  }

};

struct PlusNodeOp : public ElementBinaryNodeOp {
  template <typename ...Args>
  PlusNodeOp(Args ...args)
    : ElementBinaryNodeOp(args...) { }

  NodeOps forwardOps() {
    return {
      NodeOp(Element(_1 = _2 + _3,
                     val_,
                     child(0)->val(),
                     child(1)->val()))
    };
  }

  NodeOps backwardOps() {
    return {
      NodeOp(Add(_1, child(0)->grad(), adj_)),
      NodeOp(Add(_1, child(1)->grad(), adj_))
    };
  }

  const std::string type() {
    return "+";
  }

};

struct MinusNodeOp : public ElementBinaryNodeOp {
  template <typename ...Args>
  MinusNodeOp(Args ...args)
    : ElementBinaryNodeOp(args...) { }

  NodeOps forwardOps() {
    return {
      NodeOp(Element(_1 = _2 - _3,
                     val_, child(0)->val(), child(1)->val()))
    };
  }

  NodeOps backwardOps() {
    return {
      NodeOp(Add( _1, child(0)->grad(), adj_)),
      NodeOp(Add(-_1, child(1)->grad(), adj_))
    };
  }

  const std::string type() {
    return "-";
  }

};

struct MultNodeOp : public ElementBinaryNodeOp {
  template <typename ...Args>
  MultNodeOp(Args ...args)
    : ElementBinaryNodeOp(args...) { }

  NodeOps forwardOps() {
    return {
      NodeOp(Element(_1 = _2 * _3,
                     val_,
                     child(0)->val(),
                     child(1)->val()))
    };
  }

  NodeOps backwardOps() {
    return {
      NodeOp(Add(_1 * _2, child(0)->grad(), adj_, child(1)->val())),
      NodeOp(Add(_1 * _2, child(1)->grad(), adj_, child(0)->val()))
    };
  }

  const std::string type() {
    return "×";
  }
};

struct DivNodeOp : public ElementBinaryNodeOp {
  template <typename ...Args>
  DivNodeOp(Args ...args)
    : ElementBinaryNodeOp(args...) { }

  NodeOps forwardOps() {
    return {
      NodeOp(Element(_1 = _2 / _3,
                     val_,
                     child(0)->val(),
                     child(1)->val()))
    };
  }

  NodeOps backwardOps() {
    return {
      NodeOp(Add(_1 * 1.0f / _2,
                 child(0)->grad(),
                 adj_,
                 child(1)->val())),
      NodeOp(Add(-_1 * _2 / (_3 * _3),
                 child(1)->grad(),
                 adj_,
                 child(0)->val(),
                 child(1)->val()))
    };
  }

  const std::string type() {
    return "÷";
  }

};

// Cross-entropy node. It computes -b*log(softmax(a)), summing rowwise.
struct CrossEntropyNodeOp : public NaryNodeOp {
  template <typename ...Args>
    CrossEntropyNodeOp(Expr a, Expr b, Args ...args)
    : NaryNodeOp({a, b},
                 keywords::shape=newShape(a),
                 args...) { }

  Shape newShape(Expr a) {
    Shape shape1 = a->shape();
    shape1.set(1, 1);
    return shape1;
  }

  NodeOps forwardOps() {
    // C = sum(-logsoftmax(A) * delta(y', y))
    return {
      NodeOp(CrossEntropyPick(val_,
                              child(0)->val(),
                              child(1)->val()))
    };
  }


  NodeOps backwardOps() {
    return {
      NodeOp(CrossEntropyPickBackward(child(0)->grad(),
                                      adj_,
                                      child(0)->val(),
                                      child(1)->val()))
    };
  }


  const std::string type() {
    return "x-ent";
  }
};

struct ConcatenateNodeOp : public NaryNodeOp {
  template <typename ...Args>
  ConcatenateNodeOp(const std::vector<Expr>& nodes, Args ...args)
    : NaryNodeOp(nodes,
                 keywords::shape=newShape(nodes, keywords::Get(keywords::axis, 0, args...)),
                 args...), ax_(keywords::Get(keywords::axis, 0, args...)) { }

  Shape newShape(const std::vector<Expr>& nodes, int ax) {
    Shape shape = nodes.back()->shape();
    shape.set(ax, 0);
    for(auto child : nodes)
      shape.set(ax, shape[ax] + child->shape()[ax]);
    //std::cerr << ax << " : " << shape[0] << " " << shape[1] << std::endl;
    return shape;
  }

  void forward() {
    std::vector<Tensor> concatenees;
    for(int i = 0; i < children_.size(); ++i)
      concatenees.push_back(child(i)->val());
    Concatenate(val_, concatenees, ax_);
  }

  void backward() {
    std::vector<Tensor> deconcatenees;
    for(int i = 0; i < children_.size(); ++i) {
      auto childPtr = child(i);
      childPtr->set_zero_adjoint(); // @TODO: this is a hotfix, do this properly
      deconcatenees.push_back(childPtr->grad());
    }
    Deconcatenate(deconcatenees, adj_, ax_);
  }

  virtual size_t hash() {
    size_t seed = NaryNodeOp::hash();
    boost::hash_combine(seed, ax_);
    return seed;
  }

  const std::string type() {
    return "concat";
  }

  int ax_;
};

/*
struct TanhPlus3NodeOp : public NaryNodeOp {
  TanhPlus3NodeOp(const std::vector<Expr>& nodes)
    : NaryNodeOp(nodes, keywords::shape=newShape(nodes)) { }

  Shape newShape(const std::vector<Expr>& nodes) {
    Shape shape = nodes[0]->shape();

    for(int n = 1; n < nodes.size(); ++n) {
      Shape shapen = nodes[n]->shape();
      for(int i = 0; i < shapen.size(); ++i) {
        UTIL_THROW_IF2(shape[i] != shapen[i] && shape[i] != 1 && shapen[i] != 1,
                       "Shapes cannot be broadcasted");
        shape.set(i, std::max(shape[i], shapen[i]));
      }
    }
    return shape;
  }

  void forward() {
    Element(_1 = Tanh(_2 + _3 + _4),
            val_,
            child(0)->val(),
            child(1)->val(),
            child(2)->val());
  }

  void backward() {
    for(auto&& child : children_) {
      if(child->trainable())
        Add((1.f - _1 * _1) * _2,
            child->grad(), val_, adj_);
    }
  }

  const std::string type() {
    return "tanhPlus3";
  }

};
*/

struct AffineNodeOp : public NaryNodeOp {
  AffineNodeOp(const std::vector<Expr>& nodes)
    : NaryNodeOp(nodes, keywords::shape=newShape(nodes)) { }

  Shape newShape(const std::vector<Expr>& nodes) {
    Shape shape1 = nodes[0]->shape();
    Shape shape2 = nodes[1]->shape();
    UTIL_THROW_IF2(shape1[1] != shape2[0],
                   "matrix product requires dimensions to match");
    shape1.set(1, shape2[1]);
    return shape1;
  }

  NodeOps forwardOps() {
    return {
      NodeOp(
        Prod(std::static_pointer_cast<BackendGPU>(getBackend())->getCublasHandle(),
             val_,
             child(0)->val(),
             child(1)->val(),
             false, false);
        Add(_1, val_, child(2)->val());
      )
    };
  }

  NodeOps backwardOps() {
    // D is the adjoint, the matrix of derivatives
    // df/dA += D*B.T
    // df/dB += A.T*D
    // beta set to 1.0 in gemm, C = dot(A,B) + beta * C
    // to sum gradients from different graph parts

    return {
      NodeOp(Prod(std::static_pointer_cast<BackendGPU>(getBackend())->getCublasHandle(),
                  child(0)->grad(),
                  adj_,
                  child(1)->val(),
                  false, true, 1.0)),
      NodeOp(Prod(std::static_pointer_cast<BackendGPU>(getBackend())->getCublasHandle(),
                  child(1)->grad(),
                  child(0)->val(),
                  adj_,
                  true, false, 1.0)),
      NodeOp(Add(_1, child(2)->grad(), adj_))
    };
  }

  const std::string type() {
    return "affine";
  }
};

struct LayerNormalizationOp : public NaryNodeOp {
  LayerNormalizationOp(const std::vector<Expr>& nodes)
    : NaryNodeOp(nodes) {}

  NodeOps forwardOps() {
    return {
      NodeOp(
          LayerNormalization(val_,
                             child(0)->val(),
                             child(1)->val(),
                             (children_.size() == 3) ? child(2)->val() : nullptr))
      };
  }

  NodeOps backwardOps() {
    return {
      NodeOp(LayerNormalizationGrad(child(0)->grad(), child(1)->grad(), (children_.size() == 3) ? child(2)->grad() : nullptr,
                                    adj_, val_, child(0)->val(), child(1)->val(),
                                    (children_.size() == 3) ? child(2)->val() : nullptr))
    };
  }

  const std::string type() {
    return "layer_normalization";
  }

};


}
