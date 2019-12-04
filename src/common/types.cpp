#include "common/types.h"
#include "tensors/cpu/fbgemm/packed_gemm.h"

namespace marian {

// this function calculates the amount of bytes needed to contain a tensor of given shape and type. 
// For most situation that is trivial (just number of elements time size of single element).
// But for instance, for intransparent types like packed tensors, it cannot easily be inferred by
// multiplying. All cases are handed here and can later be passed to allocators etc. 
size_t requiredBytes(const Shape& shape, Type type) {
#if USE_FBGEMM
  if (type == Type::packed8)
  {
    int nrow, ncol;
    uint64_t packsize;
    cpu::variant::fbgemmPacked8PackInfo(shape, false, /*out=*/nrow, /*out=*/ncol, /*out=*/packsize);
    return (size_t)packsize;
  } else if (type == Type::packed16)
  {
    uint64_t packsize;
    cpu::variant::fbgemmPacked16PackInfo(shape, false, /*out=*/packsize);
    return (size_t)packsize;
  } else
#endif  // USE_FBGEMM
  {
    return shape.elements() * sizeOf(type);
  }
}

}