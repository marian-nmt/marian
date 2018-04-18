#pragma once

#include <thread>

#include "functional/functional.h"
#include "graph/node.h"
#include "tensors/gpu/cudnn_wrappers.h"
#include "tensors/tensor_operators.h"

namespace marian {

class DotNodeOp : public NaryNodeOp {
private:
  bool transA_;
  bool transB_;
  float scalar_;

public:
  DotNodeOp(Expr a, Expr b, bool transA, bool transB, float scalar)
      : NaryNodeOp({a, b}, newShape(a, b, transA, transB)),
        transA_(transA),
        transB_(transB),
        scalar_(scalar) {}

  Shape newShape(Expr a, Expr b, bool transA, bool transB) {
    auto shapeA = a->shape();
    if(transA) {
      shapeA.set(shapeA.size() - 2, a->shape()[shapeA.size() - 1]);
      shapeA.set(shapeA.size() - 1, a->shape()[shapeA.size() - 2]);
    }

    auto shapeB = b->shape();
    if(transB) {
      shapeB.set(shapeB.size() - 2, b->shape()[shapeB.size() - 1]);
      shapeB.set(shapeB.size() - 1, b->shape()[shapeB.size() - 2]);
    }

    Shape outShape = shapeA;
    outShape.set(outShape.size() - 1, shapeB[shapeB.size() - 1]);
    ABORT_IF(shapeA[shapeA.size() - 1] != shapeB[shapeB.size() - 2],
             "matrix product requires dimensions to match");
    return outShape;
  }

  NodeOps forwardOps() {
    // C = alpha * dot(op(A), op(B))
    return {NodeOp(Prod(val_,
                        child(0)->val(),
                        child(1)->val(),
                        transA_,
                        transB_,
                        0.f,
                        scalar_))};
  }

  NodeOps backwardOps() {
    // D is the adjoint, the matrix of derivatives
    // df/dA += alpha * dot(D, op(B).T)
    // df/dB += alpha * dot(op(A).T, D)
    // beta set to 1.0 in gemm, C = alpha * dot(op(A), op(B)) + beta * C
    // to sum gradients from different graph parts

    if(!transA_ && transB_)
      return {NodeOp(Prod(child(0)->grad(),
                          adj_,
                          child(1)->val(),
                          false,
                          false,
                          1.0,
                          scalar_)),
              NodeOp(Prod(child(1)->grad(),
                          adj_,
                          child(0)->val(),
                          true,
                          false,
                          1.0,
                          scalar_))};

    if(transA_ && !transB_)
      return {NodeOp(Prod(child(0)->grad(),
                          child(1)->val(),
                          adj_,
                          false,
                          true,
                          1.0,
                          scalar_)),
              NodeOp(Prod(child(1)->grad(),
                          child(0)->val(),
                          adj_,
                          false,
                          false,
                          1.0,
                          scalar_))};

    if(transA_ && transB_)
      return {NodeOp(Prod(child(0)->grad(),
                          child(1)->val(),
                          adj_,
                          true,
                          true,
                          1.0,
                          scalar_)),
              NodeOp(Prod(child(1)->grad(),
                          adj_,
                          child(0)->val(),
                          true,
                          true,
                          1.0,
                          scalar_))};

    return {NodeOp(Prod(child(0)->grad(),
                        adj_,
                        child(1)->val(),
                        false,
                        true,
                        1.0,
                        scalar_)),
            NodeOp(Prod(child(1)->grad(),
                        child(0)->val(),
                        adj_,
                        true,
                        false,
                        1.0,
                        scalar_))};
  }

  const std::string type() { return "•"; }

  const std::string color() { return "orange"; }
};

class AffineNodeOp : public NaryNodeOp {
private:
  bool transA_;
  bool transB_;
  float scalar_;

public:
  AffineNodeOp(const std::vector<Expr>& nodes,
               bool transA,
               bool transB,
               float scalar)
      : NaryNodeOp(nodes, newShape(nodes[0], nodes[1], transA, transB)),
        transA_(transA),
        transB_(transB),
        scalar_(scalar) {}

  Shape newShape(Expr a, Expr b, bool transA, bool transB) {
    auto shapeA = a->shape();
    if(transA) {
      shapeA.set(shapeA.size() - 2, a->shape()[shapeA.size() - 1]);
      shapeA.set(shapeA.size() - 1, a->shape()[shapeA.size() - 2]);
    }

    auto shapeB = b->shape();
    if(transB) {
      shapeB.set(shapeB.size() - 2, b->shape()[shapeB.size() - 1]);
      shapeB.set(shapeB.size() - 1, b->shape()[shapeB.size() - 2]);
    }

    Shape outShape = shapeA;
    outShape.set(outShape.size() - 1, shapeB[shapeB.size() - 1]);
    ABORT_IF(shapeA[shapeA.size() - 1] != shapeB[shapeB.size() - 2],
             "matrix product requires dimensions to match");
    return outShape;
  }

  NodeOps forwardOps() {
    using namespace functional;
    return {
      NodeOp(ProdWithBias(val_,
                          child(0)->val(),
                          child(1)->val(),
                          child(2)->val(),
                          transA_,
                          transB_,
                          0.f,
                          scalar_))
    };
  }

  NodeOps backwardOps() {
    // D is the adjoint, the matrix of derivatives
    // df/dA += alpha * dot(D, op(B).T)
    // df/dB += alpha * dot(op(A).T, D)
    // beta set to 1.0 in gemm, C = alpha * dot(op(A), op(B)) + beta * C
    // to sum gradients from different graph parts
    using namespace functional;

    if(!transA_ && transB_)
      return {NodeOp(Prod(child(0)->grad(),
                          adj_,
                          child(1)->val(),
                          false,
                          false,
                          1.0,
                          scalar_)),
              NodeOp(Prod(child(1)->grad(),
                          adj_,
                          child(0)->val(),
                          true,
                          false,
                          1.0,
                          scalar_)),
              NodeOp(Add(_1, child(2)->grad(), adj_))};

    if(transA_ && !transB_)
      return {NodeOp(Prod(child(0)->grad(),
                          child(1)->val(),
                          adj_,
                          false,
                          true,
                          1.0,
                          scalar_)),
              NodeOp(Prod(child(1)->grad(),
                          child(0)->val(),
                          adj_,
                          false,
                          false,
                          1.0,
                          scalar_)),
              NodeOp(Add(_1, child(2)->grad(), adj_))};

    if(transA_ && transB_)
      return {NodeOp(Prod(child(0)->grad(),
                          child(1)->val(),
                          adj_,
                          true,
                          true,
                          1.0,
                          scalar_)),
              NodeOp(Prod(child(1)->grad(),
                          adj_,
                          child(0)->val(),
                          true,
                          true,
                          1.0,
                          scalar_)),
              NodeOp(Add(_1, child(2)->grad(), adj_))};

    return {NodeOp(Prod(child(0)->grad(),
                        adj_,
                        child(1)->val(),
                        false,
                        true,
                        1.0,
                        scalar_)),
            NodeOp(Prod(child(1)->grad(),
                        child(0)->val(),
                        adj_,
                        true,
                        false,
                        1.0,
                        scalar_)),
            NodeOp(Add(_1, child(2)->grad(), adj_))};
  }

  const std::string type() { return "affine"; }
};

class DotBatchedNodeOp : public NaryNodeOp {
private:
  bool transA_;
  bool transB_;
  float scalar_;

public:
  DotBatchedNodeOp(Expr a, Expr b, bool transA, bool transB, float scalar)
      : NaryNodeOp({a, b}, newShape(a, b, transA, transB)),
        transA_(transA),
        transB_(transB),
        scalar_(scalar) {}

  Shape newShape(Expr a, Expr b, bool transA, bool transB) {
    auto shapeA = a->shape();
    if(transA) {
      shapeA.set(-2, a->shape()[-1]);
      shapeA.set(-1, a->shape()[-2]);
    }

    auto shapeB = b->shape();
    if(transB) {
      shapeB.set(-2, b->shape()[-1]);
      shapeB.set(-1, b->shape()[-2]);
    }

    Shape outShape = shapeA;
    outShape.set(-1, shapeB[-1]);
    ABORT_IF(shapeA[-1] != shapeB[-2],
             "matrix product requires dimensions to match");
    return outShape;
  }

  NodeOps forwardOps() {
    // C = alpha * dot(op(A), op(B))
    return {NodeOp(ProdBatched(val_,
                               child(0)->val(),
                               child(1)->val(),
                               transA_,
                               transB_,
                               0.f,
                               scalar_))};
  }

  NodeOps backwardOps() {
    // D is the adjoint, the matrix of derivatives
    // df/dA += alpha * dot(D, op(B).T)
    // df/dB += alpha * dot(op(A).T, D)
    // beta set to 1.0 in gemm, C = alpha * dot(op(A), op(B)) + beta * C
    // to sum gradients from different graph parts

    if(!transA_ && transB_)
      return {NodeOp(ProdBatched(child(0)->grad(),
                                 adj_,
                                 child(1)->val(),
                                 false,
                                 false,
                                 1.0,
                                 scalar_)),
              NodeOp(ProdBatched(child(1)->grad(),
                                 adj_,
                                 child(0)->val(),
                                 true,
                                 false,
                                 1.0,
                                 scalar_))};

    if(transA_ && !transB_)
      return {NodeOp(ProdBatched(child(0)->grad(),
                                 child(1)->val(),
                                 adj_,
                                 false,
                                 true,
                                 1.0,
                                 scalar_)),
              NodeOp(ProdBatched(child(1)->grad(),
                                 child(0)->val(),
                                 adj_,
                                 false,
                                 false,
                                 1.0,
                                 scalar_))};

    if(transA_ && transB_)
      return {NodeOp(ProdBatched(child(0)->grad(),
                                 child(1)->val(),
                                 adj_,
                                 true,
                                 true,
                                 1.0,
                                 scalar_)),
              NodeOp(ProdBatched(child(1)->grad(),
                                 adj_,
                                 child(0)->val(),
                                 true,
                                 true,
                                 1.0,
                                 scalar_))};

    return {NodeOp(ProdBatched(child(0)->grad(),
                               adj_,
                               child(1)->val(),
                               false,
                               true,
                               1.0,
                               scalar_)),
            NodeOp(ProdBatched(child(1)->grad(),
                               child(0)->val(),
                               adj_,
                               true,
                               false,
                               1.0,
                               scalar_))};
  }

  const std::string type() { return "•"; }

  const std::string color() { return "orange"; }
};

struct ScalarProductNodeOp : public NaryNodeOp {
  template <typename... Args>
  ScalarProductNodeOp(Expr a, Expr b, Args... args)
      : NaryNodeOp({a, b}, newShape(a, b, args...)) {}

  template <typename... Args>
  Shape newShape(Expr a, Expr b, Args... args) {
    int ax = keywords::Get(keywords::axis, -1, args...);

    Shape full = Shape::broadcast({a, b});
    ax = full.axis(ax);

    full.set(ax, 1);
    return full;
  }

  NodeOps forwardOps() {
    using namespace functional;

    return {NodeOp(Reduce(_1 * _2, val_, child(0)->val(), child(1)->val()))};
  }

  NodeOps backwardOps() {
    using namespace functional;

    return {NodeOp(Add(_1 * _2, child(0)->grad(), child(1)->val(), adj_)),
            NodeOp(Add(_1 * _2, child(1)->grad(), child(0)->val(), adj_))};
  }

  const std::string type() { return "scalar-product"; }

  const std::string color() { return "orange"; }
};

struct ElementBinaryNodeOp : public NaryNodeOp {
  ElementBinaryNodeOp(Expr a, Expr b) : NaryNodeOp({a, b}, newShape(a, b)) {}

  Shape newShape(Expr a, Expr b) { return Shape::broadcast({a, b}); }

  const std::string color() { return "yellow"; }
};

struct PlusNodeOp : public ElementBinaryNodeOp {
  PlusNodeOp(Expr a, Expr b) : ElementBinaryNodeOp(a, b) {}

  NodeOps forwardOps() {
    using namespace functional;

    return {
        NodeOp(Element(_1 = _2 + _3, val_, child(0)->val(), child(1)->val()))};
  }

  NodeOps backwardOps() {
    using namespace functional;

    return {NodeOp(Add(_1, child(0)->grad(), adj_)),
            NodeOp(Add(_1, child(1)->grad(), adj_))};
  }

  const std::string type() { return "+"; }
};

struct MinusNodeOp : public ElementBinaryNodeOp {
  MinusNodeOp(Expr a, Expr b) : ElementBinaryNodeOp(a, b) {}

  NodeOps forwardOps() {
    using namespace functional;

    return {
        NodeOp(Element(_1 = _2 - _3, val_, child(0)->val(), child(1)->val()))};
  }

  NodeOps backwardOps() {
    using namespace functional;

    return {NodeOp(Add(_1, child(0)->grad(), adj_)),
            NodeOp(Add(-_1, child(1)->grad(), adj_))};
  }

  const std::string type() { return "-"; }
};

struct MultNodeOp : public ElementBinaryNodeOp {
  MultNodeOp(Expr a, Expr b) : ElementBinaryNodeOp(a, b) {}

  NodeOps forwardOps() {
    using namespace functional;

    return {
        NodeOp(Element(_1 = _2 * _3, val_, child(0)->val(), child(1)->val()))};
  }

  NodeOps backwardOps() {
    using namespace functional;

    return {NodeOp(Add(_1 * _2, child(0)->grad(), adj_, child(1)->val())),
            NodeOp(Add(_1 * _2, child(1)->grad(), adj_, child(0)->val()))};
  }

  const std::string type() { return "×"; }
};

struct DivNodeOp : public ElementBinaryNodeOp {
  DivNodeOp(Expr a, Expr b) : ElementBinaryNodeOp(a, b) {}

  NodeOps forwardOps() {
    using namespace functional;

    return {
        NodeOp(Element(_1 = _2 / _3, val_, child(0)->val(), child(1)->val()))};
  }

  NodeOps backwardOps() {
    using namespace functional;

    return {
        NodeOp(Add(_1 * 1.0f / _2, child(0)->grad(), adj_, child(1)->val())),
        NodeOp(Add(-_1 * _2 / (_3 * _3),
                   child(1)->grad(),
                   adj_,
                   child(0)->val(),
                   child(1)->val()))};
  }

  const std::string type() { return "÷"; }
};

// struct PowNodeOp : public ElementBinaryNodeOp {
// public:
//  template <typename... Args>
//  PowNodeOp(Args... args) : ElementBinaryNodeOp(args...) {}
//
//  NodeOps forwardOps() {
//    return {NodeOp(Element(_1 = Pow(_2, _3), val_,
//                           child(0)->val(), child(1)->val()))};
//  }
//
//  NodeOps backwardOps() {
//    return {
//      NodeOp(Add(_2 * Pow(_1, _2 - 1.f) * _3,
//                 child(0)->grad(), child(0)->val(), child(1)->val(), adj_)),
//      NodeOp(Add(Pow(_1, _2) * Log(_1) * _3,
//                 child(1)->grad(), child(0)->val(), child(1)->val(), adj_))
//
//    };
//  }
//
//  const std::string type() { return "pow"; }
//};

// Cross-entropy node. It computes -b*log(softmax(a)), summing rowwise.
struct CrossEntropyNodeOp : public NaryNodeOp {
  CrossEntropyNodeOp(Expr a, Expr b) : NaryNodeOp({a, b}, newShape(a)) {}

  Shape newShape(Expr a) {
    Shape shape1 = a->shape();
    shape1.set(a->shape().size() - 1, 1);
    return shape1;
  }

  NodeOps forwardOps() {
    // C = sum(-logsoftmax(A) * delta(y', y))
    return {NodeOp(CrossEntropyPick(val_, child(0)->val(), child(1)->val()))};
  }

  NodeOps backwardOps() {
    return {NodeOp(CrossEntropyPickBackward(
        child(0)->grad(), adj_, child(0)->val(), child(1)->val()))};
  }

  const std::string type() { return "x-ent"; }
};

struct ConcatenateNodeOp : public NaryNodeOp {
  template <typename... Args>
  ConcatenateNodeOp(const std::vector<Expr>& nodes, Args... args)
      : NaryNodeOp(nodes,
                   newShape(nodes, keywords::Get(keywords::axis, 0, args...))) {
  }

  Shape newShape(const std::vector<Expr>& nodes, int ax) {
    Shape shape = nodes.back()->shape();
    ax_ = shape.axis(ax);

    int sum = 0;
    for(auto child : nodes)
      sum += child->shape()[ax_];
    shape.set(ax_, sum);

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
      childPtr
          ->set_zero_adjoint();  // @TODO: this is a hotfix, do this properly
      deconcatenees.push_back(childPtr->grad());
    }
    Deconcatenate(deconcatenees, adj_, ax_);
  }

  virtual size_t hash() {
    size_t seed = NaryNodeOp::hash();
    boost::hash_combine(seed, ax_);
    return seed;
  }

  virtual bool equal(Expr node) {
    if(!NaryNodeOp::equal(node))
      return false;
    auto cnode = std::dynamic_pointer_cast<ConcatenateNodeOp>(node);
    if(!cnode)
      return false;
    if(ax_ != cnode->ax_)
      return false;
    return true;
  }

  const std::string type() { return "concat"; }

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
        ABORT_IF(shape[i] != shapen[i] && shape[i] != 1 && shapen[i] != 1,
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

struct LayerNormalizationOp : public NaryNodeOp {
public:
  LayerNormalizationOp(const std::vector<Expr>& nodes, float eps = 1e-9)
      : NaryNodeOp(nodes), eps_(eps) {}

  NodeOps forwardOps() {
    return {NodeOp(
        LayerNormalization(val_,
                           child(0)->val(),
                           child(1)->val(),
                           (children_.size() == 3) ? child(2)->val() : nullptr,
                           eps_))};
  }

  NodeOps backwardOps() {
    return {NodeOp(LayerNormalizationGrad(
        child(0)->grad(),
        child(1)->grad(),
        (children_.size() == 3) ? child(2)->grad() : nullptr,
        adj_,
        val_,
        child(0)->val(),
        child(1)->val(),
        (children_.size() == 3) ? child(2)->val() : nullptr,
        eps_))};
  }

  const std::string type() { return "layer_normalization"; }

private:
  float eps_;
};

struct HighwayNodeOp : public NaryNodeOp {
  HighwayNodeOp(const std::vector<Expr>& nodes) : NaryNodeOp(nodes) {}

  NodeOps forwardOps() {
    return {NodeOp(HighwayForward(
        val_, child(0)->val(), child(1)->val(), child(2)->val()))};
  }

  NodeOps backwardOps() {
    return {NodeOp(HighwayBackward(child(0)->grad(),
                                   child(1)->grad(),
                                   child(2)->grad(),
                                   child(0)->val(),
                                   child(1)->val(),
                                   child(2)->val(),
                                   adj_))};
  }

  const std::string type() { return "highway"; }
};

class ConvolutionOp : public NaryNodeOp {
public:
  ConvolutionOp(const std::vector<Expr>& nodes,
                int hPad = 0,
                int wPad = 0,
                int hStride = 1,
                int wStride = 1)
      : NaryNodeOp(nodes),
        conv_(nodes[1]->shape(),
              nodes[2]->shape(),
              hPad,
              wPad,
              hStride,
              wStride) {
    conv_.getOutputShape(nodes[0]->shape(), shape_);
  }

  NodeOps forwardOps() {
    return {NodeOp(conv_.forward(
        child(0)->val(), child(1)->val(), child(2)->val(), val_))};
  }

  NodeOps backwardOps() {
    return {NodeOp(conv_.backward(child(0)->val(),
                                  child(0)->grad(),
                                  child(1)->val(),
                                  child(1)->grad(),
                                  child(2)->grad(),
                                  adj_))};
  }

  const std::string type() { return "layer_convolution"; }

protected:
  ConvolutionWrapper conv_;
};
}
