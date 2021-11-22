#pragma once

#include "graph/node.h"
#include "packed_gemm.h"
#include "tensors/cpu/integer_common.h"

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
// @TODO: don't use using namespace ...; in header files. Just don't. [UG]
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

// Pack a matrix (fp16) into cache utilization efficient way (block format) together with quantization into fp16
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
struct FbgemmPacked16PackNodeOp : public UnaryNodeOp {
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

  FbgemmPacked16PackNodeOp(Expr a, PackMatrix packMat, bool transpose)
      : UnaryNodeOp(a, newShape(a, transpose), Type::uint8),
        packMat_(packMat),
        transpose_(transpose) {
    if(packMat != PackMatrix::B)
      ABORT("Only prepacking of B (weight matrix) is supported");
    if(!memoize_)
      ABORT("Only constant weight node can be packed");
  }

  NodeOps forwardOps() override {
#if USE_FBGEMM
    return {NodeOp(fbgemmPacked16Pack(val_,
                                      child(0)->val()->data(),
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
#else // USE_FBGEMM
    ABORT("FbgemmPacked16PackNodeOp can only be used with FBGEMM enabled.");
    return { NodeOp(0) };
#endif  // USE_FBGEMM
  }

  NodeOps backwardOps() override {
    ABORT("FbgemmPacked16PackNodeOp only available for inference");
    return {NodeOp(0)};
  }

  const std::string type() override { return "packMatFp16"; }

  Shape newShape(Expr a, bool transpose) {
#if USE_FBGEMM
    auto shapeMat = a->shape();
    // Should be 2D - weight matrix
    ABORT_IF(shapeMat.size() != 2,
             "Weight Matrix should be 2D");
    fbgemmPacked16PackInfo(shapeMat,
                           transpose,
                           nrow_,
                           ncol_,
                           kernel_ncol_blocks_,
                           brow_,
                           bcol_,
                           last_brow_,
                           nbrow_,
                           nbcol_,
                           packsize_);

    Shape outShape({(int)packsize_});
    return outShape;
#else
    a; transpose;
    ABORT("Packed GEMM requires a build with USE_FBGEMM enabled");
    return Shape();
#endif  // USE_FBGEMM
  }
};

// Pack a matrix (int8) into cache utilization efficient way (block format) together with quantization into int8
// PackMatrix packMat_: the type of packed matrix - A or B matrix
// marian::Type packType_: the type the input matrix is packed - packed8avx2 or packed8avx512
// bool transpose_: transpose
// int nrow_: the number of rows
// int ncol_: the number of columns
// uint64_t packsize_: the size of the packed matrix
//                    (the size of int8 packed B from fbgemm:PackAWithQuantRowOffset + quantization scale, offset and zero point)
struct FbgemmPacked8PackNodeOp : public UnaryNodeOp {
  PackMatrix packMat_;
  marian::Type packType_;
  bool transpose_;
  int nrow_;
  int ncol_;
  uint64_t packsize_;
  float quantizeRange_;

  FbgemmPacked8PackNodeOp(Expr a,
                          PackMatrix packMat,
                          marian::Type packType,
                          bool transpose,
                          float quantizeRange)
      : UnaryNodeOp(a, newShape(a, packType, transpose), Type::uint8),
        packMat_(packMat),
        packType_(packType),
        transpose_(transpose),
        quantizeRange_(quantizeRange){
    if(packMat != PackMatrix::B)
      ABORT("Only prepacking of B (weight matrix) is supported");
    if(!memoize_)
      ABORT("Only constant weight node can be packed");
  }

  NodeOps forwardOps() override {
#if USE_FBGEMM
    return {NodeOp(fbgemmPacked8Pack(val_,
                                     child(0)->val()->data(),
                                     packType_,
                                     transpose_,
                                     nrow_,
                                     ncol_,
                                     packsize_,
                                     quantizeRange_))
    };
#else // USE_FBGEMM
    ABORT("FbgemmPacked8PackNodeOp can only be used with FBGEMM enabled.");
    return { NodeOp(0) };
#endif  // USE_FBGEMM
  }

  NodeOps backwardOps() override {
    ABORT("FbgemmPacked8PackNodeOp only available for inference");
    return {NodeOp(0)};
  }

  const std::string type() override { return "packMatInt8"; }

#if USE_FBGEMM
  Shape newShape(Expr a, marian::Type packType, bool transpose) {
    fbgemmPacked8PackInfo(
        a->shape(),
        packType,
        transpose,
        nrow_,
        ncol_,
        packsize_);
    Shape outShape({(int)packsize_});
    return outShape;
  }
#else
  Shape newShape(Expr /*a*/, marian::Type /*packType*/, bool /*transpose*/) {
    ABORT("Packed GEMM requires a build with USE_FBGEMM enabled");
    return Shape();
  }
#endif  // USE_FBGEMM
};


// Affine transform (matrix multiplication) using packed B matrix
// float scalar_: scalar multiplier
// size_t m_: the number of rows in A and C
// size_t n_: the number of columns in B and C
// size_t k_: the number of columns in A and the number of rows in C
// bool transA_: transpose A
// bool transB_: transpose B
class FbgemmPacked16AffineNodeOp : public NaryNodeOp {
private:
  size_t m_;
  size_t n_;
  size_t k_;
  bool transA_;
  bool transB_;

public:
  FbgemmPacked16AffineNodeOp(const std::vector<Expr>& nodes, Shape bShape, bool transA, bool transB, float /*scalar*/)
    : NaryNodeOp(nodes, newShape(nodes[0], bShape, transA, transB), Type::float32)/*, scalar_(scalar)*/ {
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
#if USE_FBGEMM
    return {
      NodeOp(fbgemmPacked16Gemm(val_,
                                child(0)->val(),
                                child(1)->val(),
                                children().size() > 2 ? child(2)->val() : nullptr, // pass only if it has a bias
                                m_,
                                n_,
                                transA_))
    };
#else // USE_FBGEMM
    ABORT("FbgemmPacked16AffineNodeOp can only be used with FBGEMM enabled.");
    return { NodeOp(0) };
#endif  // USE_FBGEMM
  }

  NodeOps backwardOps() override {
    ABORT("Only used for inference");
    return {NodeOp(0)};
  }

  const std::string type() override { return "gemmPacked16"; }
};

// Affine transform (matrix multiplication) using packed B matrix
// Especially, this gemm performs quantized gemms in 8-bit integers.
// float scalar_: scalar multiplier
// size_t m_: the number of rows in A and C
// size_t n_: the number of columns in B and C
// size_t k_: the number of columns in A and the number of rows in C
// bool transA_: transpose A
// bool transB_: transpose B
class FbgemmPacked8AffineNodeOp : public NaryNodeOp {
private:
  size_t m_;
  size_t n_;
  size_t k_;
  bool transA_;
  bool transB_;

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-private-field"
#endif

  Type elementType_;

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif


public:
  FbgemmPacked8AffineNodeOp(Type elementType,
                            const std::vector<Expr>& nodes,
                            Shape bShape,
                            bool transA,
                            bool transB,
                            float /*scalar*/)
      : NaryNodeOp(nodes, newShape(nodes[0], bShape, transA, transB), Type::float32),
        elementType_(elementType) {
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
    NodeOps nodeOps;
#if USE_FBGEMM
    // Do addBias only if it has a bias term
    if (children().size() > 2) {
      nodeOps = { NodeOp(fbgemmPacked8Gemm(elementType_,
                                           val_,
                                           child(0)->val(),
                                           child(1)->val(),
                                           m_,
                                           n_,
                                           k_,
                                           transA_,
                                           transB_);
                       marian::cpu::integer::AddBias(val_, child(2)->val())) };
    } else {
      nodeOps = { NodeOp(fbgemmPacked8Gemm(elementType_,
                                           val_,
                                           child(0)->val(),
                                           child(1)->val(),
                                           m_,
                                           n_,
                                           k_,
                                           transA_,
                                           transB_)) };
    }
#else // USE_FBGEMM
    ABORT("FbgemmPacked8AffineNodeOp can only be used with FBGEMM enabled.");
#endif  // USE_FBGEMM

    return nodeOps;
  }

  NodeOps backwardOps() override {
    ABORT("Only used for inference");
    return {NodeOp(0)};
  }

  const std::string type() override { return "gemmPacked8"; }
};

static inline Expr affine(Type elementType,
                          Expr a,
                          Expr b,
                          Shape bShape,
                          Expr c,
                          bool transA,
                          bool transB,
                          float scalar) {
  std::vector<Expr> nodes = {a, b, c};

  if (elementType == Type::packed16)
    return Expression<FbgemmPacked16AffineNodeOp>(nodes, bShape, transA, transB, scalar);
  else if (isPacked(elementType) && sizeOf(elementType) == 1)
    return Expression<cpu::variant::FbgemmPacked8AffineNodeOp>(
        elementType, nodes, bShape, transA, transB, scalar);
  else {
    ABORT("Only int8 and fp16 are available. {}", elementType);
    return nullptr;
  }
}

static inline Expr pack(Type elementType, Expr a, PackMatrix packMat, bool transpose, float quantizeRange = 0.f) {
  if (elementType == Type::packed16)
    return Expression<FbgemmPacked16PackNodeOp>(a, packMat, transpose);
  else if (isPacked(elementType) && sizeOf(elementType) == 1)
    return Expression<cpu::variant::FbgemmPacked8PackNodeOp>(a, packMat, elementType, transpose, quantizeRange);
  else {
    ABORT("Only int8 and fp16 are available. {}", elementType);
    return nullptr;
  }
}

static inline Expr dot(Type elementType, Expr a, Expr b, Shape bShape, bool transA, bool transB, float scalar) {
  std::vector<Expr> nodes = {a, b};

  if (elementType == Type::packed16)
    return Expression<FbgemmPacked16AffineNodeOp>(nodes, bShape, transA, transB, scalar);
  else if (isPacked(elementType) && sizeOf(elementType) == 1)
    return Expression<cpu::variant::FbgemmPacked8AffineNodeOp>(
        elementType, nodes, bShape, transA, transB, scalar);
  else {
    ABORT("Only int8 and fp16 are available. {}", elementType);
    return nullptr;
  }
}

}  // namespace variant
}  // namespace cpu
}  // namespace marian
