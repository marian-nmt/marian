#include "common/types.h"
#include "tensors/cpu/sharp/packed_gemm.h"

namespace marian {

// this function calculates the amount of bytes needed to contain a tensor of given shape and type. 
// For most situation that is trivial (just number of elements time size of single element).
// But for instance, for intransparent types like packed tensors, it cannot easily be inferred by
// multiplying. All cases are handed here and can later be passed to allocators etc. 
size_t requiredBytes(const Shape& shape, Type type) {
  if(isPacked(type)) {
    uint64_t packsize;
    cpu::variant::PackInfoFp32(shape, false, packsize);
    return (size_t)packsize;
  } else {
    return shape.elements() * sizeOf(type);
  }
}

}