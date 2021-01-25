#include "common/types.h"
#include "tensors/cpu/fbgemm/packed_gemm.h"

namespace marian {

// this function calculates the amount of bytes needed to contain a tensor of given shape and type. 
// For most situation that is trivial (just number of elements time size of single element).
// But for instance, for intransparent types like packed tensors, it cannot easily be inferred by
// multiplying. All cases are handed here and can later be passed to allocators etc. 
size_t requiredBytes(const Shape& shape, Type type) {
#if USE_FBGEMM
  if (isPacked(type)) {
    if (sizeOf(type) == 1) {
      // Type::packed8avx2 || type == Type::packed8avx512
      // AVX2 and AVX512 CPUs have different cache and vector lanes,
      // so the optimal memory layouts for them are different.
      int nrow, ncol;
      uint64_t packsize;
      cpu::variant::fbgemmPacked8PackInfo(shape, type, false, /*out=*/nrow, /*out=*/ncol, /*out=*/packsize);
      return (size_t)packsize;
    } else if (type == Type::packed16) {
      uint64_t packsize;
      cpu::variant::fbgemmPacked16PackInfo(shape, false, /*out=*/packsize);
      return (size_t)packsize;
    } else {
      ABORT("Not a supported data type: {}", type);
      return 0;
    }
  }
#endif  // USE_FBGEMM 

  if (isIntgemm(type)) {
    /* Intgemm tensors have an extra float at the back that stores the quantization multiplier */
    return shape.elements() * sizeOf(type) + sizeOf(Type::float32);
  } else {
    return shape.elements() * sizeOf(type);
  }
  
}

} // namespace marian