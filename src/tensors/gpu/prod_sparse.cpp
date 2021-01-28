#ifdef _MSC_VER
#pragma warning(disable: 4505) // warning C4505: '__float2half_rz': unreferenced local function has been removed (missing 'static inline')
#endif

#include <cublas_v2.h>
#include <cusparse.h>

// clang-format off
#include "tensors/gpu/prod.h"
#include "tensors/gpu/backend.h"
#include "tensors/gpu/cuda_helpers.h"
// clang-format on

// what a nightmare
#if CUDA_VERSION >= 11000
#include "tensors/gpu/prod_sparse_cu11.h"
#else
#include "tensors/gpu/prod_sparse_cu10.h"
#endif

namespace marian {
namespace gpu {

void CSRProd(marian::Tensor C,
             Ptr<Allocator> allocator,
             const marian::Tensor& S_values,
             const marian::Tensor& S_indices,
             const marian::Tensor& S_offsets,
             const marian::Tensor& D,
             bool transS,
             bool swapOperands,
             float beta) {
  if(S_values->type() == Type::float32 && D->type() == Type::float32) {
    TypedSparseGemm</*ElementType=*/float>::CSRProd(C, allocator, S_values, S_indices, S_offsets, D, transS, swapOperands, beta);
#if COMPILE_FP16
  } else if(S_values->type() == Type::float16 && D->type() == Type::float16) {
    TypedSparseGemm</*ElementType=*/half>::CSRProd(C, allocator, S_values, S_indices, S_offsets, D, transS, swapOperands, (half)beta);
#endif
  } else {
    ABORT("Types {} and {} are not supported for sparse GEMM operations", S_values->type(), D->type());
  }
}

}
}