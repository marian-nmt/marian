#pragma once

#include <thread>

#include "common/hash.h"
#include "functional/functional.h"
#include "graph/node.h"
#include "tensors/tensor_operators.h"

#ifdef CUDNN
#include "tensors/gpu/cudnn_wrappers.h"
#endif

namespace marian {

class LambdaNodeOp : public NaryNodeOp {
private:
  typedef const std::vector<Expr>& Inputs;
  typedef std::function<void(Expr, Inputs)> LambdaNodeFunctor;
  
  std::unique_ptr<LambdaNodeFunctor> forward_;
  std::unique_ptr<LambdaNodeFunctor> backward_;
  
  size_t externalHash_;

public:
  LambdaNodeOp(Inputs inputs, Shape shape, Type type, 
               LambdaNodeFunctor forward, 
               size_t externalHash = 0)
  : NaryNodeOp(inputs, shape, type), 
    forward_(new LambdaNodeFunctor(forward)),
    externalHash_(externalHash) {
    Node::trainable_ = !!backward_;
  }

  LambdaNodeOp(Inputs inputs, Shape shape, Type type, 
               LambdaNodeFunctor forward,
               LambdaNodeFunctor backward, 
               size_t externalHash = 0) 
  : NaryNodeOp(inputs, shape, type), 
    forward_(new LambdaNodeFunctor(forward)),
    backward_(new LambdaNodeFunctor(backward)),
    externalHash_(externalHash) {
  }

  void forward() override {
    (*forward_)(this, children_);
  }

  void backward() override {
    ABORT_IF(!backward_, "No backward lambda given?");
    (*backward_)(this, children_);
  }

  const std::string type() override { return "lambda"; }

  virtual size_t hash() override {
    size_t seed = NaryNodeOp::hash();
    if(externalHash_ != 0) {
      util::hash_combine(seed, externalHash_);
    } else {
      util::hash_combine(seed, forward_.get());
      util::hash_combine(seed, backward_.get());
    }
    return seed;
  }

  virtual bool equal(Expr node) override {
    if(!NaryNodeOp::equal(node))
      return false;
    auto cnode = std::dynamic_pointer_cast<LambdaNodeOp>(node);
    if(!cnode)
      return false;
    if(forward_ != cnode->forward_)   // pointer compare on purpose
      return false;
    if(backward_ != cnode->backward_) // pointer compare on purpose
      return false;
    return true;
  }
};

class DotNodeOp : public NaryNodeOp {
private:
  friend class SerializationHelpers;
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
      shapeB.set(shapeB.size() - 2, b->shape()[shapeB.size() - 1]); // @TODO: why not use negative indices?
      shapeB.set(shapeB.size() - 1, b->shape()[shapeB.size() - 2]);
    }

    Shape outShape = shapeA;
    outShape.set(outShape.size() - 1, shapeB[shapeB.size() - 1]);
    ABORT_IF(shapeA[shapeA.size() - 1] != shapeB[shapeB.size() - 2],
             "Matrix product requires inner dimensions to match in {}{} * {}{}", std::string(shapeA), transA, std::string(shapeB), transB);
    return outShape;
  }

  NodeOps forwardOps() override {
    // C = alpha * dot(op(A), op(B))
    return {NodeOp(Prod(val_,
                        child(0)->val(),
                        child(1)->val(),
                        transA_,
                        transB_,
                        0.f,
                        scalar_))};
  }

  NodeOps backwardOps() override {
    // D is the adjoint, the matrix of derivatives
    // df/dA += alpha * dot(D, op(B).T)
    // df/dB += alpha * dot(op(A).T, D)
    // beta set to 1.0 in gemm, C = alpha * dot(op(A), op(B)) + beta * C
    // to sum gradients from different graph parts
    
    auto isParameter = [](Expr p) {
      return std::dynamic_pointer_cast<ParamNode>(p) != nullptr;
    };

    // if child A is not a parameter (i.e. activations) use computeType float32 for accumulation
    Type computeTypeA = child(0)->trainable() ? child(0)->grad()->type() : Type::float32;
    if(!isParameter(child(0)) && computeTypeA == Type::float16)
      computeTypeA = Type::float32;

    // if child B is not a parameter (i.e. activations) use computeType float32 for accumulation
    Type computeTypeB = child(1)->trainable() ? child(1)->grad()->type() : Type::float32;
    if(!isParameter(child(1)) && computeTypeB == Type::float16)
      computeTypeB = Type::float32;

    if(!transA_ && transB_)
      return {NodeOp(Prod(child(0)->grad(),
                          adj_,
                          child(1)->val(),
                          false,
                          false,
                          1.0,
                          scalar_, computeTypeA)),
              NodeOp(Prod(child(1)->grad(),
                          adj_,
                          child(0)->val(),
                          true,
                          false,
                          1.0,
                          scalar_, computeTypeB))};

    if(transA_ && !transB_)
      return {NodeOp(Prod(child(0)->grad(),
                          child(1)->val(),
                          adj_,
                          false,
                          true,
                          1.0,
                          scalar_, computeTypeA)),
              NodeOp(Prod(child(1)->grad(),
                          child(0)->val(),
                          adj_,
                          false,
                          false,
                          1.0,
                          scalar_, computeTypeB))};

    if(transA_ && transB_)
      return {NodeOp(Prod(child(0)->grad(),
                          child(1)->val(),
                          adj_,
                          true,
                          true,
                          1.0,
                          scalar_, computeTypeA)),
              NodeOp(Prod(child(1)->grad(),
                          adj_,
                          child(0)->val(),
                          true,
                          true,
                          1.0,
                          scalar_, computeTypeB))};

    return {NodeOp(Prod(child(0)->grad(),
                        adj_,
                        child(1)->val(),
                        false,
                        true,
                        1.0,
                        scalar_, computeTypeA)),
            NodeOp(Prod(child(1)->grad(),
                        child(0)->val(),
                        adj_,
                        true,
                        false,
                        1.0,
                        scalar_, computeTypeB))};
  }

  const std::string type() override { return "dot"; }

  virtual size_t hash() override {
    size_t seed = NaryNodeOp::hash();
    util::hash_combine(seed, transA_);
    util::hash_combine(seed, transB_);
    util::hash_combine(seed, scalar_);
    return seed;
  }

  virtual bool equal(Expr node) override {
    if(!NaryNodeOp::equal(node))
      return false;
    auto cnode = std::dynamic_pointer_cast<DotNodeOp>(node);
    if(!cnode)
      return false;
    if(transA_ != cnode->transA_)
      return false;
    if(transB_ != cnode->transB_)
      return false;
    if(scalar_ != cnode->scalar_)
      return false;
    return true;
  }

  const std::string color() override { return "orange"; }
};

class AffineNodeOp : public NaryNodeOp {
private:
  friend class SerializationHelpers;
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
             "Matrix product requires inner dimensions to match in {}{} * {}{}", std::string(shapeA), transA, std::string(shapeB), transB);
    return outShape;
  }

  NodeOps forwardOps() override {
    using namespace functional;
    
    return {
      NodeOp(Affine(val_,
                    graph()->allocator(),
                    child(0)->val(),
                    child(1)->val(),
                    child(2)->val(),
                    transA_,
                    transB_,
                    0.f,
                    scalar_,
                    /*doRelu=*/false))
    };
  }

  NodeOps backwardOps() override {
    // D is the adjoint, the matrix of derivatives
    // df/dA += alpha * dot(D, op(B).T)
    // df/dB += alpha * dot(op(A).T, D)
    // beta set to 1.0 in gemm, C = alpha * dot(op(A), op(B)) + beta * C
    // to sum gradients from different graph parts

    auto isParameter = [](Expr p) {
      return std::dynamic_pointer_cast<ParamNode>(p) != nullptr;
    };

    // if child A is not a parameter (i.e. activations) use computeType float32 for accumulation
    Type computeTypeA = child(0)->trainable() ? child(0)->grad()->type() : Type::float32;
    if(!isParameter(child(0)) && computeTypeA == Type::float16)
      computeTypeA = Type::float32;

    // if child B is not a parameter (i.e. activations) use computeType float32 for accumulation
    Type computeTypeB = child(1)->trainable() ? child(1)->grad()->type() : Type::float32;
    if(!isParameter(child(1)) && computeTypeB == Type::float16)
      computeTypeB = Type::float32;

    // if child C (bias) is not a parameter (i.e. activations) use computeType float32 for accumulation
    Type computeTypeC = child(2)->trainable() ? child(2)->grad()->type() : Type::float32;
    if(!isParameter(child(2)) && computeTypeC == Type::float16)
      computeTypeC = Type::float32;

    // We reduce bias gradients with a matrix multiply
    if(!transA_ && transB_)
      return {
          NodeOp(Prod(child(0)->grad(),
                      adj_,
                      child(1)->val(),
                      false,
                      false,
                      1.0,
                      scalar_, computeTypeA)),
          NodeOp(Prod(child(1)->grad(),
                      adj_,
                      child(0)->val(),
                      true,
                      false,
                      1.0,
                      scalar_, computeTypeB)),
          NodeOp(Prod(child(2)->grad(), child(3)->val(), adj_, true, false, 0.f, 1.f, computeTypeC))
      };

    if(transA_ && !transB_)
      return {
          NodeOp(Prod(child(0)->grad(),
                      child(1)->val(),
                      adj_,
                      false,
                      true,
                      1.0,
                      scalar_, computeTypeA)),
          NodeOp(Prod(child(1)->grad(),
                      child(0)->val(),
                      adj_,
                      false,
                      false,
                      1.0,
                      scalar_, computeTypeB)),
          NodeOp(Prod(child(2)->grad(), child(3)->val(), adj_, true, false, 0.f, 1.f, computeTypeC))
      };

    if(transA_ && transB_)
      return {
          NodeOp(Prod(child(0)->grad(),
                      child(1)->val(),
                      adj_,
                      true,
                      true,
                      1.0,
                      scalar_, computeTypeA)),
          NodeOp(Prod(child(1)->grad(),
                      adj_,
                      child(0)->val(),
                      true,
                      true,
                      1.0,
                      scalar_, computeTypeB)),
          NodeOp(Prod(child(2)->grad(), child(3)->val(), adj_, true, false, 0.f, 1.f, computeTypeC))
      };

    return {
        NodeOp(Prod(child(0)->grad(),
                    adj_,
                    child(1)->val(),
                    false,
                    true,
                    1.0,
                    scalar_, computeTypeA)),
        NodeOp(Prod(child(1)->grad(),
                    child(0)->val(),
                    adj_,
                    true,
                    false,
                    1.0,
                    scalar_, computeTypeB)),
        NodeOp(Prod(child(2)->grad(), child(3)->val(), adj_, true, false, 0.f, 1.f, computeTypeC))
    };
  }

  const std::string type() override { return "affine"; }

  virtual size_t hash() override {
    size_t seed = NaryNodeOp::hash();
    util::hash_combine(seed, transA_);
    util::hash_combine(seed, transB_);
    util::hash_combine(seed, scalar_);
    return seed;
  }

  virtual bool equal(Expr node) override {
    if(!NaryNodeOp::equal(node))
      return false;
    auto cnode = std::dynamic_pointer_cast<AffineNodeOp>(node);
    if(!cnode)
      return false;
    if(transA_ != cnode->transA_)
      return false;
    if(transB_ != cnode->transB_)
      return false;
    if(scalar_ != cnode->scalar_)
      return false;
    return true;
  }

};

class AffineWithReluNodeOp : public NaryNodeOp {
private:
  friend class SerializationHelpers;
  bool transA_;
  bool transB_;
  float scalar_;

public:
  AffineWithReluNodeOp(Expr a, 
                       Expr b, 
                       Expr bias,
                       bool transA,
                       bool transB,
                       float scalar)
      : NaryNodeOp({a, b, bias}, newShape(a, b, transA, transB)),
        transA_(transA),
        transB_(transB),
        scalar_(scalar) {
    ABORT_IF(!graph()->isInference() || graph()->getDeviceId().type != DeviceType::gpu,
             "AffineWithReluNodeOp currently only supported for inference on GPU");
  }

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
             "Matrix product requires inner dimensions to match in {}{} * {}{}", std::string(shapeA), transA, std::string(shapeB), transB);
    return outShape;
  }

  NodeOps forwardOps() override {
    ABORT_IF(!graph()->isInference() || graph()->getDeviceId().type != DeviceType::gpu,
             "AffineWithReluNodeOp currently only supported for inference on GPU");
    
    return {
      NodeOp(Affine(val_,
                    graph()->allocator(),
                    child(0)->val(),
                    child(1)->val(),
                    child(2)->val(),
                    transA_,
                    transB_,
                    0.f,
                    scalar_,
                    /*doRelu=*/true))
    };
  }

  NodeOps backwardOps() override {
    ABORT("AffineWithReluNodeOp cannot be used for training??");
    return {};
  }

  const std::string type() override { return "affineWithRelu"; }

  virtual size_t hash() override {
    size_t seed = NaryNodeOp::hash();
    util::hash_combine(seed, transA_);
    util::hash_combine(seed, transB_);
    util::hash_combine(seed, scalar_);
    return seed;
  }

  virtual bool equal(Expr node) override {
    if(!NaryNodeOp::equal(node))
      return false;
    auto cnode = std::dynamic_pointer_cast<AffineWithReluNodeOp>(node);
    if(!cnode)
      return false;
    if(transA_ != cnode->transA_)
      return false;
    if(transB_ != cnode->transB_)
      return false;
    if(scalar_ != cnode->scalar_)
      return false;
    return true;
  }
};

class DotBatchedNodeOp : public NaryNodeOp {
private:
  friend class SerializationHelpers;
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

    ABORT_IF(shapeA[-1] != shapeB[-2],
             "Batched matrix product requires inner dimensions to match in {}{} * {}{}",
             std::string(shapeA), transA, std::string(shapeB), transB);
    
    // create shapes for batch dimensions only
    auto shapeBatchA = shapeA;
    shapeBatchA.set(-1, 1);
    shapeBatchA.set(-2, 1);
    
    auto shapeBatchB = shapeB;
    shapeBatchB.set(-1, 1);
    shapeBatchB.set(-2, 1);

    // broadcast batch dimensions
    auto shapeOut = Shape::broadcast({shapeBatchA, shapeBatchB});

    // set non-batch dimensions in output
    shapeOut.set(-2, shapeA[-2]);
    shapeOut.set(-1, shapeB[-1]);
    
    return shapeOut;
  }

  NodeOps forwardOps() override {
    // C = alpha * dot(op(A), op(B))
    return {NodeOp(ProdBatched(val_,
                               graph()->allocator(),
                               child(0)->val(),
                               child(1)->val(),
                               transA_,
                               transB_,
                               0.f,
                               scalar_))};
  }

  NodeOps backwardOps() override {
    // D is the adjoint, the matrix of derivatives
    // df/dA += alpha * dot(D, op(B).T)
    // df/dB += alpha * dot(op(A).T, D)
    // beta set to 1.0 in gemm, C = alpha * dot(op(A), op(B)) + beta * C
    // to sum gradients from different graph parts

    if(!transA_ && transB_)
      return {NodeOp(ProdBatched(child(0)->grad(),
                                 graph()->allocator(),
                                 adj_,
                                 child(1)->val(),
                                 false,
                                 false,
                                 1.0,
                                 scalar_)),
              NodeOp(ProdBatched(child(1)->grad(),
                                 graph()->allocator(),
                                 adj_,
                                 child(0)->val(),
                                 true,
                                 false,
                                 1.0,
                                 scalar_))};

    if(transA_ && !transB_)
      return {NodeOp(ProdBatched(child(0)->grad(),
                                 graph()->allocator(),
                                 child(1)->val(),
                                 adj_,
                                 false,
                                 true,
                                 1.0,
                                 scalar_)),
              NodeOp(ProdBatched(child(1)->grad(),
                                 graph()->allocator(),
                                 child(0)->val(),
                                 adj_,
                                 false,
                                 false,
                                 1.0,
                                 scalar_))};

    if(transA_ && transB_)
      return {NodeOp(ProdBatched(child(0)->grad(),
                                 graph()->allocator(),
                                 child(1)->val(),
                                 adj_,
                                 true,
                                 true,
                                 1.0,
                                 scalar_)),
              NodeOp(ProdBatched(child(1)->grad(),
                                 graph()->allocator(),
                                 adj_,
                                 child(0)->val(),
                                 true,
                                 true,
                                 1.0,
                                 scalar_))};

    return {NodeOp(ProdBatched(child(0)->grad(),
                               graph()->allocator(),
                               adj_,
                               child(1)->val(),
                               false,
                               true,
                               1.0,
                               scalar_)),
            NodeOp(ProdBatched(child(1)->grad(),
                               graph()->allocator(),
                               child(0)->val(),
                               adj_,
                               true,
                               false,
                               1.0,
                               scalar_))};
  }

  const std::string type() override { return "bdot"; }

  virtual size_t hash() override {
    size_t seed = NaryNodeOp::hash();
    util::hash_combine(seed, transA_);
    util::hash_combine(seed, transB_);
    util::hash_combine(seed, scalar_);
    return seed;
  }

  virtual bool equal(Expr node) override {
    if(!NaryNodeOp::equal(node))
      return false;
    auto cnode = std::dynamic_pointer_cast<DotBatchedNodeOp>(node);
    if(!cnode)
      return false;
    if(transA_ != cnode->transA_)
      return false;
    if(transB_ != cnode->transB_)
      return false;
    if(scalar_ != cnode->scalar_)
      return false;
    return true;
  }

  const std::string color() override { return "orange"; }
};

class DotBatchedLegacyNodeOp : public NaryNodeOp {
private:
  friend class SerializationHelpers;
  bool transA_;
  bool transB_;
  float scalar_;

public:
  DotBatchedLegacyNodeOp(Expr a, Expr b, bool transA, bool transB, float scalar)
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
             "Batched matrix product requires inner dimensions to match in {}{} * {}{}", std::string(shapeA), transA, std::string(shapeB), transB);
    return outShape;
  }

  NodeOps forwardOps() override {
    // C = alpha * dot(op(A), op(B))
    return {NodeOp(ProdBatchedLegacy(val_,
                                     graph()->allocator(),
                                     child(0)->val(),
                                     child(1)->val(),
                                     transA_,
                                     transB_,
                                     0.f,
                                     scalar_))};
  }

  NodeOps backwardOps() override {
    // D is the adjoint, the matrix of derivatives
    // df/dA += alpha * dot(D, op(B).T)
    // df/dB += alpha * dot(op(A).T, D)
    // beta set to 1.0 in gemm, C = alpha * dot(op(A), op(B)) + beta * C
    // to sum gradients from different graph parts

    if(!transA_ && transB_)
      return {NodeOp(ProdBatchedLegacy(child(0)->grad(),
                                       graph()->allocator(),
                                       adj_,
                                       child(1)->val(),
                                       false,
                                       false,
                                       1.0,
                                       scalar_)),
              NodeOp(ProdBatchedLegacy(child(1)->grad(),
                                       graph()->allocator(),
                                       adj_,
                                       child(0)->val(),
                                       true,
                                       false,
                                       1.0,
                                       scalar_))};
    if(transA_ && !transB_)
      return {NodeOp(ProdBatchedLegacy(child(0)->grad(),
                                       graph()->allocator(),
                                       child(1)->val(),
                                       adj_,
                                       false,
                                       true,
                                       1.0,
                                       scalar_)),
              NodeOp(ProdBatchedLegacy(child(1)->grad(),
                                 graph()->allocator(),
                                 child(0)->val(),
                                 adj_,
                                 false,
                                 false,
                                 1.0,
                                 scalar_))};
    if(transA_ && transB_)
      return {NodeOp(ProdBatchedLegacy(child(0)->grad(),
                                       graph()->allocator(),
                                       child(1)->val(),
                                       adj_,
                                       true,
                                       true,
                                       1.0,
                                       scalar_)),
              NodeOp(ProdBatchedLegacy(child(1)->grad(),
                                       graph()->allocator(),
                                       adj_,
                                       child(0)->val(),
                                       true,
                                       true,
                                       1.0,
                                       scalar_))};
    return {NodeOp(ProdBatchedLegacy(child(0)->grad(),
                                     graph()->allocator(),
                                     adj_,
                                     child(1)->val(),
                                     false,
                                     true,
                                     1.0,
                                     scalar_)),
            NodeOp(ProdBatchedLegacy(child(1)->grad(),
                                     graph()->allocator(),
                                     child(0)->val(),
                                     adj_,
                                     true,
                                     false,
                                     1.0,
                                     scalar_))};
  }

  const std::string type() override { return "bdot_legacy"; }

  virtual size_t hash() override {
    size_t seed = NaryNodeOp::hash();
    util::hash_combine(seed, transA_);
    util::hash_combine(seed, transB_);
    util::hash_combine(seed, scalar_);
    return seed;
  }

  virtual bool equal(Expr node) override {
    if(!NaryNodeOp::equal(node))
      return false;
    auto cnode = std::dynamic_pointer_cast<DotBatchedLegacyNodeOp>(node);
    if(!cnode)
      return false;
    if(transA_ != cnode->transA_)
      return false;
    if(transB_ != cnode->transB_)
      return false;
    if(scalar_ != cnode->scalar_)
      return false;
    return true;
  }

  const std::string color() override { return "orange"; }
};

// Note: To reduce code duplication, we use the same NodeOp for C = op(S) x D and C = D x op(S).
// Set swapOperands to select the latter.
class CSRDotNodeOp : public NaryNodeOp {
  bool transS_;
  bool swapOperands_;
public:
  CSRDotNodeOp(const Shape& S_shape, Expr S_values, Expr S_indices,
               Expr S_offsets, Expr D, bool transS, bool swapOperands)
    : NaryNodeOp({ S_values, S_indices, S_offsets, D },
                 newShape(S_shape, S_values, S_indices, S_offsets, D, transS, swapOperands),
                 NaryNodeOp::commonType({S_values, D})),
      transS_(transS), swapOperands_(swapOperands) {
    matchOrAbort<IndexType>(S_indices->value_type());
    matchOrAbort<IndexType>(S_offsets->value_type());

    ABORT_IF(swapOperands_, "Implementation for this is wonky, if you use this tell us.");
  }

  Shape newShape(const Shape& S_shape, Expr S_values, Expr S_indices, Expr S_offsets, Expr D, bool transS, bool swapOperands) {
    ABORT_IF(S_values->shape().size() != 1 || S_indices->shape().size() != 1 || S_offsets->shape().size() != 1,
        "Sparse matrix components must all be vectors");
    ABORT_IF(S_values->shape() != S_indices->shape(),
        "Sparse matrix values and indices must have the same shape");
    ABORT_IF(S_shape.size() != 2,
        "Sparse matrix must have rank 2");
    ABORT_IF(S_offsets->shape()[0] - 1 != S_shape[0],
        "Sparse matrix offset vector has incorrect size");
    auto outShape = D->shape();
    ABORT_IF(S_shape[transS == swapOperands ? 1 : 0] != outShape[-(int)swapOperands],
             "Matrix product requires inner dimensions to match");
    outShape.set(-(int)swapOperands, S_shape[transS != swapOperands]);
    return outShape;
  }

  NodeOps forwardOps() override {
    return {NodeOp(CSRProd(val_,
                           graph()->allocator(),
                           child(0)->val(), child(1)->val(), child(2)->val(),
                           child(3)->val(),
                           /*transS=*/transS_, /*swapOperands=*/swapOperands_, /*beta=*/0))};
  }

  NodeOps backwardOps() override {
    return { nullptr, // can't backprop into the sparse matrix (the gradient is dense)
             nullptr,
             nullptr,
             NodeOp(CSRProd(child(3)->grad(), // child(3) = D
                            graph()->allocator(),
                            child(0)->val(), child(1)->val(), child(2)->val(), // children(0..2) = A
                            adj_,
                            /*transS=*/!transS_, /*swapOperands=*/swapOperands_, /*beta=*/1))};
  }

  const std::string type() override { return "csr_dot"; }

  virtual size_t hash() override {
    size_t seed = NaryNodeOp::hash();
    for(auto s : shape())
      util::hash_combine(seed, s);
    util::hash_combine(seed, transS_);
    util::hash_combine(seed, swapOperands_);
    return seed;
  }

  virtual bool equal(Expr node) override {
    if(!NaryNodeOp::equal(node))
      return false;
    auto cnode = std::dynamic_pointer_cast<CSRDotNodeOp>(node);
    if(!cnode)
      return false;
    if(transS_ != cnode->transS_)
      return false;
    if(shape() != cnode->shape())
      return false;
    if(swapOperands_ != cnode->swapOperands_)
      return false;
    return true;
  }

  const std::string color() override { return "orange"; }
};

struct ScalarProductNodeOp : public NaryNodeOp {
  ScalarProductNodeOp(Expr a, Expr b, int axis)
      : NaryNodeOp({a, b}, newShape(a, b, axis)) {}

  Shape newShape(Expr a, Expr b, int axis) {
    Shape full = Shape::broadcast({a, b});
    axis_ = full.axis(axis);

    full.set(axis_, 1);
    return full;
  }

  NodeOps forwardOps() override {
    using namespace functional;

    return {NodeOp(Reduce(_1 * _2, val_, child(0)->val(), child(1)->val()))};
  }

  NodeOps backwardOps() override {
    using namespace functional;

    return {NodeOp(Add(_1 * _2, child(0)->grad(), child(1)->val(), adj_)),
            NodeOp(Add(_1 * _2, child(1)->grad(), child(0)->val(), adj_))};
  }

  const std::string type() override { return "scalar-product"; }

  const std::string color() override { return "orange"; }

  virtual size_t hash() override {
    size_t seed = NaryNodeOp::hash();
    util::hash_combine(seed, axis_);
    return seed;
  }

  virtual bool equal(Expr node) override {
    if(!NaryNodeOp::equal(node))
      return false;
    auto cnode = std::dynamic_pointer_cast<ScalarProductNodeOp>(node);
    if(!cnode)
      return false;
    if(axis_ != cnode->axis_)
      return false;
    return true;
  }

  int axis_;
};

struct RowsNodeOp : public NaryNodeOp {
  RowsNodeOp(Expr a, Expr indices)
    : NaryNodeOp({a, indices}, newShape(a, indices), a->value_type()) {
      matchOrAbort<IndexType>(indices->value_type());
  }

  NodeOps forwardOps() override {
    return {NodeOp(
        CopyRows(val_, child(0)->val(), child(1)->val()))};
  }

  NodeOps backwardOps() override {
    return {NodeOp(PasteRows(child(0)->grad(), adj_, child(1)->val()))};
  }

  Shape newShape(Expr a, Expr indices) {
    Shape shape = a->shape();
    ABORT_IF(shape.size() != 2,
             "rows operator can only be used with 2-dimensional tensors");
    shape.set(0, (int)indices->shape().elements());
    return shape;
  }

  const std::string type() override { return "rows"; }

  const std::string color() override { return "orange"; }
};

// This operation gathers elements of a tensor along an axis.
// This is like PyTorch gather().
// For example, this can be used for:
//  - Same index applied to all batch items:
//    'index' has 1 in the axes that match batch axes in the input, and axis set to the one axis that gets selected over.
//    Example: Selecting Transformer head 0, i.e. return a[:,1,:,:]
//      axis = -3
//      a  : (B,  H , S, T)     B=batch dim, H=#heads, S=src length, T=trg length
//      idx: (   #1#, 1, 1)     #1# denotes 'axis'. All values are zero.
//      out: (B,  1 , S, T)     out[b, 0, s, t] == a[b, idx[/*0,*/ 0, s, t], s, t]
//  - Same data with batched indices (today's rows()):
//    'data' has 1 in the batch axes.
//    Example: Embedding lookup as done today using rows():
//      axis = -2
//      e  : (     V , E)        V=vocab size, E=embedding dimension
//      idx: (#(B*S)#, 1)        B=batch size, S=source length, idx values are in range 0..V-1
//      out: ( (B*S) , E)        out[b, s, e] == e[/*0,*/ idx[b, s, 0], e]
//  - Batched selection (x-ent scenario): Both 'index' and 'data' have matching batch axes.
//    Example: Cross-entropy loss as -gather(logSoftmax(logits), groundTruth, axis=-1):
//      axis = -1
//      lp : (B, T,  V )        B=batch size, T=trg length, V=vocab size
//      idx: (B, T, #1#)        idx values are in range 0..V-1
//      out: (B, T,  1 )        out[b,t,0] == lp[b, t, idx[b, t, 0]]
// Example for 2D tensor with axis=0:
//  | t[index[0, 0] 0]   t[index[0, 1] 1] |
//  | t[index[1, 0] 0]   t[index[1, 1] 1] |
// And for axis 1:
//  | t[0 index[0, 0]]   t[0 index[0, 1]] |
//  | t[1 index[1, 0]]   t[1 index[1, 1]] |
// For a 3-D tensor the output is specified by:
//  out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
//  out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
//  out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2
// 'a' and 'indices' must have the same rank.
struct GatherNodeOp : public NaryNodeOp {
  GatherNodeOp(Expr a, int axis, Expr indices)
      : NaryNodeOp({a, indices}, newShape(a, axis, indices), a->value_type()),
        axis_(a->shape().axis(axis)) {
    matchOrAbort<IndexType>(indices->value_type());
  }

  NodeOps forwardOps() override {
    return {NodeOp(
      // @TODO: rename to gather
      Select(val_, child(0)->val(), child(1)->val(), axis_))};
  }

  NodeOps backwardOps() override {
    return {NodeOp(
      // @TODO: rename to scatter
      Insert</*add=*/true>(child(0)->grad(), adj_, child(1)->val(), axis_))};
  }

  Shape newShape(Expr a, int axis, Expr indices) {
    Shape shape = a->shape();
    axis = shape.axis(axis);
    auto rank = shape.size();
    ABORT_IF(rank != indices->shape().size(), "Mismatching ranks for input ({}) and indices ({})", std::string(shape), std::string(indices->shape()));
    shape.set(axis, indices->shape()[axis]);
    for (size_t i = 0; i < rank; ++i) {
      if (i != axis) {
        ABORT_IF(indices->shape()[i] != shape[i] && indices->shape()[i] != 1,
            "Dimensions must match or broadcast for input ({}) and indices ({})", std::string(shape), std::string(indices->shape()));
      }
    }
    return shape;
  }

  const std::string type() override { return "gather"; }

  const std::string color() override { return "orange"; }

  virtual size_t hash() override {
    if(!hash_) {
      size_t seed = NaryNodeOp::hash();
      util::hash_combine(seed, axis_);
      hash_ = seed;
    }
    return hash_;
  }

  virtual bool equal(Expr node) override {
    if(!NaryNodeOp::equal(node))
      return false;
    auto cnode = std::dynamic_pointer_cast<GatherNodeOp>(node);
    if(!cnode)
      return false;
    if(axis_ != cnode->axis_)
      return false;
    return true;
  }

private:
  friend class SerializationHelpers;
  int axis_;
};

struct ScatterNodeOp : public NaryNodeOp {
  ScatterNodeOp(Expr a, int axis, Expr indices, Expr source)
      : NaryNodeOp({a, indices, source}, newShape(a, axis, indices, source), a->value_type()),
        axis_(a->shape().axis(axis)) {
    matchOrAbort<IndexType>(indices->value_type());
  }

  NodeOps forwardOps() override {
    return {NodeOp(
      CopyCast(val_, child(0)->val()); // @TODO: use normal copy
      Insert</*add=*/false>(val_, child(2)->val(), child(1)->val(), axis_)
    )};
  }

  NodeOps backwardOps() override {
    ABORT("backward for ScatterNodeOp not yet implemented");
  }

  Shape newShape(Expr a, int axis, Expr indices, Expr source) {
    ABORT_IF(axis != -1, "only last dimensions");
    ABORT_IF(indices->shape() != source->shape(), "Shapes must match");

    Shape shape = a->shape();
    // @TODO: do proper checking
    return shape;
  }

  const std::string type() override { return "scatter"; }

  const std::string color() override { return "orange"; }

  virtual size_t hash() override {
    if(!hash_) {
      size_t seed = NaryNodeOp::hash();
      util::hash_combine(seed, axis_);
      hash_ = seed;
    }
    return hash_;
  }

  virtual bool equal(Expr node) override {
    if(!NaryNodeOp::equal(node))
      return false;
    auto cnode = std::dynamic_pointer_cast<ScatterNodeOp>(node);
    if(!cnode)
      return false;
    if(axis_ != cnode->axis_)
      return false;
    return true;
  }

private:
  friend class SerializationHelpers;
  int axis_;
};

struct ColsNodeOp : public NaryNodeOp {
  ColsNodeOp(Expr a, Expr indices)
    : NaryNodeOp({a, indices}, newShape(a, indices), a->value_type()) {
    matchOrAbort<IndexType>(indices->value_type());
  }

  NodeOps forwardOps() override {
    return {NodeOp(CopyCols(val_, child(0)->val(), child(1)->val()))};
  }

  NodeOps backwardOps() override {
    return {NodeOp(PasteCols(child(0)->grad(), adj_, child(1)->val()))};
  }

  Shape newShape(Expr a, Expr indices) {
    Shape shape = a->shape();
    shape.set(1, (int)indices->shape().elements());
    return shape;
  }

  const std::string type() override { return "cols"; }

  const std::string color() override { return "orange"; }
};


struct ElementBinaryNodeOp : public NaryNodeOp {
  ElementBinaryNodeOp(Expr a, Expr b)
   : NaryNodeOp({a, b}, newShape(a, b)) {}

  Shape newShape(Expr a, Expr b) { return Shape::broadcast({a, b}); }

  const std::string color() override { return "yellow"; }
};

struct PlusNodeOp : public ElementBinaryNodeOp {
  PlusNodeOp(Expr a, Expr b) : ElementBinaryNodeOp(a, b) {}

  NodeOps forwardOps() override {
    using namespace functional;

    return {
        NodeOp(Element(_1 = _2 + _3, val_, child(0)->val(), child(1)->val()))};
  }

  NodeOps backwardOps() override {
    using namespace functional;

    return {NodeOp(Add(_1, child(0)->grad(), adj_)),
            NodeOp(Add(_1, child(1)->grad(), adj_))};
  }

  const std::string type() override { return "+"; }
};

struct MinusNodeOp : public ElementBinaryNodeOp {
  MinusNodeOp(Expr a, Expr b) : ElementBinaryNodeOp(a, b) {}

  NodeOps forwardOps() override {
    using namespace functional;

    return {
        NodeOp(Element(_1 = _2 - _3, val_, child(0)->val(), child(1)->val()))};
  }

  NodeOps backwardOps() override {
    using namespace functional;

    return {NodeOp(Add(_1, child(0)->grad(), adj_)),
            NodeOp(Add(-_1, child(1)->grad(), adj_))};
  }

  const std::string type() override { return "-"; }
};

struct MultNodeOp : public ElementBinaryNodeOp {
  MultNodeOp(Expr a, Expr b) : ElementBinaryNodeOp(a, b) {}

  NodeOps forwardOps() override {
    using namespace functional;

    return {
        NodeOp(Element(_1 = _2 * _3, val_, child(0)->val(), child(1)->val()))};
  }

  NodeOps backwardOps() override {
    using namespace functional;

    return {NodeOp(Add(_1 * _2, child(0)->grad(), adj_, child(1)->val())),
            NodeOp(Add(_1 * _2, child(1)->grad(), adj_, child(0)->val()))};
  }

  const std::string type() override { return "*"; }
};

struct DivNodeOp : public ElementBinaryNodeOp {
  DivNodeOp(Expr a, Expr b) : ElementBinaryNodeOp(a, b) {}

  NodeOps forwardOps() override {
    using namespace functional;

    return {
        NodeOp(Element(_1 = _2 / _3, val_, child(0)->val(), child(1)->val()))};
  }

  NodeOps backwardOps() override {
    using namespace functional;

    return {
        NodeOp(Add(_1 * 1.0f / _2, child(0)->grad(), adj_, child(1)->val())),
        NodeOp(Add(-_1 * _2 / (_3 * _3),
                   child(1)->grad(),
                   adj_,
                   child(0)->val(),
                   child(1)->val()))};
  }

  const std::string type() override { return "/"; }
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

struct LogAddExpNodeOp : public ElementBinaryNodeOp {
  LogAddExpNodeOp(Expr a, Expr b) : ElementBinaryNodeOp(a, b) {}

  NodeOps forwardOps() override {
    using namespace functional;
    return {NodeOp(Element(
        _1 = logaddexp(_2, _3), val_, child(0)->val(), child(1)->val()))};
  }

  NodeOps backwardOps() override {
    using namespace functional;

    // d/dx (ln( exp(x) + (exp(y)) = exp(x) / (exp(x) + exp(y)) = 1 / (1 +
    // exp(y-x)) = sigmoid(x-y)
    return {NodeOp(Add(_1 * sigmoid(_2 - _3),
                       child(0)->grad(),
                       adj_,
                       child(0)->val(),
                       child(1)->val())),
            NodeOp(Add(_1 * sigmoid(_3 - _2),
                       child(1)->grad(),
                       adj_,
                       child(0)->val(),
                       child(1)->val()))};
  }

  // TODO: this is not a "type" (as in data type). It's an operator name.
  const std::string type() override { return "logaddexp"; }
};

struct MaximumNodeOp : public ElementBinaryNodeOp {
  MaximumNodeOp(Expr a, Expr b) : ElementBinaryNodeOp(a, b) {}

  NodeOps forwardOps() override {
    using namespace functional;
    return {NodeOp(
        Element(_1 = max(_2, _3), val_, child(0)->val(), child(1)->val()))};
  }

  NodeOps backwardOps() override {
    using namespace functional;

    return {NodeOp(Add((_2 > _3) * _1,
                       child(0)->grad(),
                       adj_,
                       child(0)->val(),
                       child(1)->val())),
            NodeOp(Add((_2 <= _3) * _1,
                       child(1)->grad(),
                       adj_,
                       child(0)->val(),
                       child(1)->val()))};
  }

  const std::string type() override { return "max"; }
};

// TODO: lotsa code dup here!
struct MinimumNodeOp : public ElementBinaryNodeOp {
  MinimumNodeOp(Expr a, Expr b) : ElementBinaryNodeOp(a, b) {}

  NodeOps forwardOps() override {
    using namespace functional;
    return {NodeOp(
        Element(_1 = min(_2, _3), val_, child(0)->val(), child(1)->val()))};
  }

  NodeOps backwardOps() override {
    using namespace functional;

    return {NodeOp(Add((_2 < _3) * _1,
                       child(0)->grad(),
                       adj_,
                       child(0)->val(),
                       child(1)->val())),
            NodeOp(Add((_2 >= _3) * _1,
                       child(1)->grad(),
                       adj_,
                       child(0)->val(),
                       child(1)->val()))};
  }

  const std::string type() override { return "min"; }
};

struct CmpNodeOp : public ElementBinaryNodeOp {
  CmpNodeOp(Expr a, Expr b, int cmp_, bool not_) : ElementBinaryNodeOp(a, b), cmp_(cmp_), not_(not_) {
    //setTrainable(false); // has no gradient
    // Note: ^^ Disabled because it currently causing Marian to choke, for unknown reasons.
    //       Not setting this will not change the result since the vector of gradient functions is empty.
  }

  NodeOps forwardOps() override {
    using namespace functional;

    return {
      NodeOp(Element(_1 = ((((_2 > _3) - (_2 < _3)) == (float)cmp_) != not_),
             val_, child(0)->val(), child(1)->val()))};
  }

  NodeOps backwardOps() override { return {}; }

  const std::string type() override {
    switch (cmp_) {
    case -1: return not_ ? "ge" : "lt";
    case  0: return not_ ? "ne" : "eq";
    case  1: return not_ ? "le" : "gt";
    }
    ABORT("Should not get here??");
  }

  virtual size_t hash() override {
    if(!hash_) {
      size_t seed = NaryNodeOp::hash();
      util::hash_combine(seed, cmp_);
      util::hash_combine(seed, not_);
      hash_ = seed;
    }
    return hash_;
  }

  virtual bool equal(Expr node) override {
    if(!NaryNodeOp::equal(node))
      return false;
    auto cnode = std::dynamic_pointer_cast<CmpNodeOp>(node);
    if(!cnode)
      return false;
    if(cmp_ != cnode->cmp_)
      return false;
    if(not_ != cnode->not_)
      return false;
    return true;
  }

private:
  int cmp_;  // -1: less; 0: equal; 1: greater
  bool not_; // invert result if true
};

// In each j-th row, take the corresponding j-th label index i from indices and compute:
// For each vocabulary item v, the only non-zero element in a row in the sum is the item
// that matches the label indexed by i (the picked element).
// C = sum_{v in V}(-logsoftmax(A) * delta(v, i) = -logsoftmax(A)[i]
class CrossEntropyNodeOp : public NaryNodeOp {
private:
  float labelSmoothingAlpha_;

public:
  CrossEntropyNodeOp(Expr a, Expr indices, float labelSmoothingAlpha, Type outputType = Type::float32)
    : NaryNodeOp({a, indices}, newShape(a), outputType),
      labelSmoothingAlpha_(labelSmoothingAlpha) {
    matchOrAbort<IndexType>(indices->value_type());
    int rows   = a->shape().elements() / a->shape()[-1];
    int labels = indices->shape().elements();
    ABORT_IF(rows != labels, "Number of examples and labels does not match: {} != {}", rows, labels);
  }

  Shape newShape(Expr a) {
    Shape shape1 = a->shape();
    shape1.set(a->shape().size() - 1, 1);
    return shape1;
  }

  NodeOps forwardOps() override {
    return {NodeOp(CrossEntropyPick(val_, child(0)->val(), child(1)->val(), labelSmoothingAlpha_))};
  }

  NodeOps backwardOps() override {
    return {NodeOp(CrossEntropyPickBackward(
        child(0)->grad(), adj_, child(0)->val(), child(1)->val(), labelSmoothingAlpha_))};
  }

  virtual size_t hash() override {
    size_t seed = NaryNodeOp::hash();
    util::hash_combine(seed, labelSmoothingAlpha_);
    return seed;
  }

  virtual bool equal(Expr node) override {
    if(!NaryNodeOp::equal(node))
      return false;
    auto cnode = std::dynamic_pointer_cast<CrossEntropyNodeOp>(node);
    if(!cnode)
      return false;
    if(labelSmoothingAlpha_ != cnode->labelSmoothingAlpha_)
      return false;
    return true;
  }

  const std::string type() override { return "x-ent"; }
};

struct ConcatenateNodeOp : public NaryNodeOp {
  ConcatenateNodeOp(const std::vector<Expr>& nodes, int axis)
      : NaryNodeOp(nodes, newShape(nodes, axis)) {
  }

  Shape newShape(const std::vector<Expr>& nodes, int ax) {
    ABORT_IF(nodes.empty(), "No child nodes given");

    Shape shape = nodes[0]->shape();
    axis_ = shape.axis(ax);

    int sum = 0;
    auto checkShape = shape;
    for(auto child : nodes) {
      checkShape.set(axis_, child->shape()[axis_]); // don't abort on different sizes on axis dim.
      ABORT_IF(checkShape != child->shape(),
               "Child shapes {} and {} cannot be concatenated along axis {}",
               shape, child->shape(), ax);

      sum += child->shape()[axis_];
    }
    shape.set(axis_, sum);

    return shape;
  }

  void forward() override {
    std::vector<Tensor> concatenees;
    for(size_t i = 0; i < children_.size(); ++i)
      concatenees.push_back(child(i)->val());
    Concatenate(val_, concatenees, axis_);
  }

  void backward() override {
    std::vector<Tensor> deconcatenees;
    for(size_t i = 0; i < children_.size(); ++i) {
      auto childPtr = child(i);
      childPtr->set_zero_adjoint();  // @TODO: this is a hotfix, do this properly
      deconcatenees.push_back(childPtr->grad());
    }
    Deconcatenate(deconcatenees, adj_, axis_);
  }

  virtual size_t hash() override {
    size_t seed = NaryNodeOp::hash();
    util::hash_combine(seed, axis_);
    return seed;
  }

  virtual bool equal(Expr node) override {
    if(!NaryNodeOp::equal(node))
      return false;
    auto cnode = std::dynamic_pointer_cast<ConcatenateNodeOp>(node);
    if(!cnode)
      return false;
    if(axis_ != cnode->axis_)
      return false;
    return true;
  }

  const std::string type() override { return "concat"; }

private:
  friend class SerializationHelpers;
  int axis_;
};

// layer norm along last axis
struct LayerNormalizationOp : public NaryNodeOp {
public:
  LayerNormalizationOp(const std::vector<Expr>& nodes, float eps = 1e-9)
      : NaryNodeOp(nodes), eps_(eps) {
    // @TODO: dimension check
  }

  NodeOps forwardOps() override {
    return {NodeOp(
        LayerNormalization(val_,
                           child(0)->val(),
                           child(1)->val(),
                           (children_.size() == 3) ? child(2)->val() : nullptr,
                           eps_))};
  }

  // @BUGBUG: backward has not been tested for broadcasting gamma/beta
  NodeOps backwardOps() override {
    return {NodeOp(
      LayerNormalizationGrad(
        graph()->allocator(),
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

  const std::string type() override { return "layer_normalization"; }

  virtual size_t hash() override {
    size_t seed = NaryNodeOp::hash();
    util::hash_combine(seed, eps_);
    return seed;
  }

  virtual bool equal(Expr node) override {
    if(!NaryNodeOp::equal(node))
      return false;
    auto cnode = std::dynamic_pointer_cast<LayerNormalizationOp>(node);
    if(!cnode)
      return false;
    if(eps_ != cnode->eps_)
      return false;
    return true;
  }

private:
  friend class SerializationHelpers; // @TODO: use the same name for this as SqrtNodeOp
  float eps_;
};

// RMS norm along last axis
struct RMSNormalizationOp : public NaryNodeOp {
public:
  RMSNormalizationOp(const std::vector<Expr>& nodes, float eps = 1e-9)
      : NaryNodeOp(nodes), eps_(eps) {
    // @TODO: dimension check
  }

  NodeOps forwardOps() override {
    return {NodeOp(
        RMSNormalization(val_,
                         child(0)->val(),
                         child(1)->val(),
                         (children_.size() == 3) ? child(2)->val() : nullptr,
                         eps_))};
  }

  // @BUGBUG: backward has not been tested for broadcasting gamma/beta
  NodeOps backwardOps() override {
    return {NodeOp(
      RMSNormalizationGrad(
        graph()->allocator(),
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

  const std::string type() override { return "rms_normalization"; }

  virtual size_t hash() override {
    size_t seed = NaryNodeOp::hash();
    util::hash_combine(seed, eps_);
    return seed;
  }

  virtual bool equal(Expr node) override {
    if(!NaryNodeOp::equal(node))
      return false;
    auto cnode = std::dynamic_pointer_cast<RMSNormalizationOp>(node);
    if(!cnode)
      return false;
    if(eps_ != cnode->eps_)
      return false;
    return true;
  }

private:
  friend class SerializationHelpers; // @TODO: use the same name for this as SqrtNodeOp
  float eps_;
};


struct HighwayNodeOp : public NaryNodeOp {
  HighwayNodeOp(const std::vector<Expr>& nodes) : NaryNodeOp(nodes) {}

  NodeOps forwardOps() override {
    return {NodeOp(HighwayForward(
        val_, child(0)->val(), child(1)->val(), child(2)->val()))};
  }

  NodeOps backwardOps() override {
    return {NodeOp(HighwayBackward(child(0)->grad(),
                                   child(1)->grad(),
                                   child(2)->grad(),
                                   child(0)->val(),
                                   child(1)->val(),
                                   child(2)->val(),
                                   adj_))};
  }

  const std::string type() override { return "highway"; }
};

#ifdef CUDNN

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

  NodeOps forwardOps() override {
    return {NodeOp(conv_.forward(
        child(0)->val(), child(1)->val(), child(2)->val(), val_))};
  }

  NodeOps backwardOps() override {
    return {NodeOp(conv_.backward(child(0)->val(),
                                  child(0)->grad(),
                                  child(1)->val(),
                                  child(1)->grad(),
                                  child(2)->grad(),
                                  adj_))};
  }

  const std::string type() override { return "layer_convolution"; }

protected:
  ConvolutionWrapper conv_;
};
#endif
}  // namespace marian
