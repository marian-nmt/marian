#pragma once

#include "graph/node.h"
#include "tensors/cpu/sharp/packed_gemm.h"

#if USE_FBGEMM

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif

#include "3rd_party/fbgemm/include/fbgemm/FbgemmFP16.h"

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

using namespace fbgemm;
#endif  // USE_FBGEMM

namespace marian {
namespace cpu {
namespace variant {

// Enumeration for the Matrix used in pack functions
// A matrix - 0, B matrix - 1
enum class PackMatrix : uint8_t {
  A = 0x00,
  B = 0x01
};

// Pack a matrix into cache utilization efficient way (block format)
// PackMatrix packMat_: the type of packed matrix - A or B matrix
// bool transpose_: transpose
// int nrow_: the number of rows
// int ncol_: the number of columns
// int kernel_ncol_blocks_: the number of column blocks
// int brow_: the number of rows in a block
// int bcol_: the number of columns in a block
// int last_brow_: the number of rows in the last block
// int nbrow_: row index in a block
// int nbcol_: column index in a block
// uint64_t packsize_: the size of the packed matrix
//                    (the number of fp16 elements + padding (1024) + extra temporary memory (256))
struct PackNodeOp : public UnaryNodeOp {
  PackMatrix packMat_;
  bool transpose_;
  int nrow_;
  int ncol_;
  int kernel_ncol_blocks_;
  int brow_;
  int bcol_;
  int last_brow_;
  int nbrow_;
  int nbcol_;
  uint64_t packsize_;

  PackNodeOp(Expr a, PackMatrix packMat, bool transpose, float clipValue)
      : UnaryNodeOp(a, newShape(a, transpose), Type::uint8),
        packMat_(packMat),
        transpose_(transpose) {
    if(packMat != PackMatrix::B)
      ABORT("Only prepacking of B (weight matrix) is supported");
    if(clipValue != 0)
      ABORT("Clipping is not supported");
    if(!memoize_)
      ABORT("Only constant weight node can be packed");
  }

  NodeOps forwardOps() override {
    return {NodeOp(PackFp32(val_,
                            child(0)->val(),
                            transpose_,
                            nrow_,
                            ncol_,
                            kernel_ncol_blocks_,
                            brow_,
                            bcol_,
                            last_brow_,
                            nbrow_,
                            nbcol_,
                            packsize_))
    };
  }

  NodeOps backwardOps() override {
    ABORT("PackNodeOp only available for inference");
    return {NodeOp(0)};
  }

  const std::string type() override { return "packMat"; }

  Shape newShape(Expr a, bool transpose) {
#if USE_FBGEMM
    auto shapeMat = a->shape();
    // Should be 2D - weight matrix
    ABORT_IF(shapeMat.size() != 2,
             "Weight Matrix should be 2D");
    nrow_ = transpose ? shapeMat[1] : shapeMat[0];
    ncol_ = transpose ? shapeMat[0] : shapeMat[1];
    kernel_ncol_blocks_ = 2;
    brow_ = 512;
    bcol_ = 8 * kernel_ncol_blocks_;
    last_brow_ = nrow_ % brow_ == 0 ? brow_ : nrow_ % brow_;
    nbrow_ = nrow_ % brow_ == 0 ? nrow_ / brow_ : (nrow_ + brow_) / brow_;
    nbcol_ = ncol_ % bcol_ == 0 ? ncol_ / bcol_ : (ncol_ + bcol_) / bcol_;
    const int padding = 1024;  // required by sw pipelined kernels
    const int specialMem = 256;
    packsize_ = ((nbrow_ * brow_) * (nbcol_ * bcol_)) * sizeof(fbgemm::float16) + padding + specialMem;

    Shape outShape({(int)packsize_});

    return outShape;
#else // USE_FBGEMM
    ABORT("Packed GEMM requires a build with USE_FBGEMM enabled");
    return Shape();
#endif  // USE_FBGEMM
  }
};

// Affine transform (matrix multiplication) using packed B matrix
// float scalar_: scalar multiplier
// size_t m_: the number of rows in A and C
// size_t n_: the number of columns in B and C
// size_t k_: the number of columns in A and the number of rows in C
// bool transA_: transpose A
// bool transB_: transpose B
class AffineNodeOp : public NaryNodeOp {
private:
  float scalar_;
  size_t m_;
  size_t n_;
  size_t k_;
  bool transA_;
  bool transB_;

public:
  AffineNodeOp(const std::vector<Expr>& nodes, Shape bShape, bool transA, bool transB, float scalar)
      : NaryNodeOp(nodes, newShape(nodes[0], bShape, transA, transB), Type::float32),
        scalar_(scalar) {
    transA_ = transA;
    transB_ = transB;
    m_ = nodes[0]->shape().elements() / nodes[0]->shape()[-1];
    k_ = nodes[0]->shape().back();
    if(transA)
      std::swap(m_, k_);

    size_t l = bShape.elements() / bShape[-1];
    n_ = bShape[-1];
    if(transB)
      std::swap(l, n_);
  }

  Shape newShape(Expr a, Shape bShape, bool transA, bool transB) {
    auto shapeA = a->shape();
    if(transA) {
      shapeA.set(shapeA.size() - 2, a->shape()[shapeA.size() - 1]);
      shapeA.set(shapeA.size() - 1, a->shape()[shapeA.size() - 2]);
    }

    auto shapeB = bShape;
    if(transB) {
      shapeB.set(shapeB.size() - 2, bShape[shapeB.size() - 1]);
      shapeB.set(shapeB.size() - 1, bShape[shapeB.size() - 2]);
    }

    Shape outShape = shapeA;
    outShape.set(outShape.size() - 1, shapeB[shapeB.size() - 1]);
    ABORT_IF(shapeA[shapeA.size() - 1] != shapeB[shapeB.size() - 2],
             "Matrix product requires inner dimensions to match");
    return outShape;
  }

  NodeOps forwardOps() override {
    return {
      NodeOp(GemmPackFp32(val_,
                          child(0)->val(),
                          child(1)->val(),
                          child(2)->val(),
                          m_,
                          n_,
                          transA_))
    };
  }

  NodeOps backwardOps() override {
    ABORT("Only used for inference");
    return {NodeOp(0)};
  }

  const std::string type() override { return "fp16packed"; }
};

static inline Expr affine(Expr a, Expr b, Shape bShape, Expr c, bool transA, bool transB, float scalar) {
  std::vector<Expr> nodes = {a, b, c};
  return Expression<cpu::variant::AffineNodeOp>(nodes, bShape, transA, transB, scalar);
}

static inline Expr pack(Expr a, PackMatrix packMat, bool transpose, float clipValue) {
  return Expression<cpu::variant::PackNodeOp>(a, packMat, transpose, clipValue);
}

}  // namespace variant
}  // namespace cpu
}  // namespace marian
