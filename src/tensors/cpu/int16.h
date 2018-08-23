#pragma once

#include "graph/node.h"
#include "tensors/cpu/sharp/int_gemm.h"

namespace marian {
namespace cpu {
namespace int16 {

struct QuantizeNodeOp : public UnaryNodeOp {
  float clipValue_;

  QuantizeNodeOp(Expr a, float clipValue)
      : UnaryNodeOp(a, Type::int16), clipValue_{clipValue} {}

  NodeOps forwardOps() override {
    return {NodeOp(Quantize16(val_, child(0)->val(), clipValue_))};
  }

  NodeOps backwardOps() override {
    ABORT("Only used for inference");
    return {NodeOp(0)};
  }

  const std::string type() override { return "quantizeInt16"; }
};

class DotNodeOp : public NaryNodeOp {
private:
  float scalar_;

public:
  DotNodeOp(Expr a, Expr b, float scalar)
      : NaryNodeOp({a, b}, newShape(a, b)), scalar_(scalar) {}

  Shape newShape(Expr a, Expr b) {
    auto shapeA = a->shape();
    auto shapeB = b->shape();

    // Computing A * B^T
    shapeB.set(-2, b->shape()[-1]);
    shapeB.set(-1, b->shape()[-2]);

    Shape outShape = shapeA;
    outShape.set(-1, shapeB[-1]);
    ABORT_IF(shapeA[-1] != shapeB[-2],
             "matrix product requires dimensions to match");
    return outShape;
  }

  NodeOps forwardOps() override {
    return {NodeOp(ProdInt16(val_, child(0)->val(), child(1)->val(), scalar_))};
  }

  NodeOps backwardOps() override {
    ABORT("Only used for inference");
    return {NodeOp(0)};
  }

  const std::string type() override { return "dotInt16"; }
};

class AffineNodeOp : public NaryNodeOp {
private:
  float scalar_;

public:
  AffineNodeOp(const std::vector<Expr>& nodes, float scalar)
      : NaryNodeOp(nodes, newShape(nodes[0], nodes[1])), scalar_(scalar) {}

  Shape newShape(Expr a, Expr b) {
    auto shapeA = a->shape();
    auto shapeB = b->shape();

    // Computing A * B^T
    shapeB.set(-2, b->shape()[-1]);
    shapeB.set(-1, b->shape()[-2]);

    Shape outShape = shapeA;
    outShape.set(-1, shapeB[-1]);
    ABORT_IF(shapeA[-1] != shapeB[-2],
             "matrix product requires dimensions to match");
    return outShape;
  }

  NodeOps forwardOps() override {
    return {
      NodeOp(ProdInt16(val_, child(0)->val(), child(1)->val(), scalar_);
             AddBias(val_, child(2)->val()))
    };
  }

  NodeOps backwardOps() override {
    ABORT("Only used for inference");
    return {NodeOp(0)};
  }

  const std::string type() override { return "affineInt16"; }
};

static inline Expr dot(Expr a, Expr b, float scalar) {
  return Expression<cpu::int16::DotNodeOp>(a, b, scalar);
}

static inline Expr affine(Expr a, Expr b, Expr c, float scalar) {
  std::vector<Expr> nodes = {a, b, c};
  return Expression<cpu::int16::AffineNodeOp>(nodes, scalar);
}

static inline Expr quantize(Expr a, float clipValue) {
  return Expression<cpu::int16::QuantizeNodeOp>(a, clipValue);
}

}  // namespace int16
}  // namespace cpu
}  // namespace marian
