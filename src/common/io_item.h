#pragma once

#include "common/shape.h"
#include "common/types.h"

#include <string>

namespace marian {
namespace io {

struct Item {
  std::vector<char> bytes;
  const char* ptr{0};
  bool mapped{false};

  std::string name;
  Shape shape;
  Type type{Type::float32};

  const char* data() const {
    if(mapped)
      return ptr;
    else
      return bytes.data();
  }

  size_t size() const { // @TODO: review this again for 256-bytes boundary alignment
    return requiredBytes(shape, type);
  }

  // Extend this item with data and shape from the input item, creating a flattened concatenation.
  void append(const Item& other) {
    ABORT_IF(mapped, "Memory-mapped items cannot be appended");
    ABORT_IF(type != other.type, "Only item of same type can be appended");

    // abort if any of the shapes is not a flat array, i.e. the number of elements in the
    // last dimension has to correspond to the number of bytes.
    ABORT_IF(shape[-1] != shape.elements(), "1 - Only flat items can be appended : {}", shape);
    ABORT_IF(other.shape[-1] != other.shape.elements(), "2 - Only flat items can be appended: {}", other.shape);

    // cut to size (get rid of padding if any) to make append operation work correctly
    size_t bytesWithoutPadding = shape.elements() * sizeOf(type);
    bytes.resize(bytesWithoutPadding);

    shape.set(-1, shape.elements() + other.shape.elements());

    size_t addbytesWithoutPadding = other.shape.elements() * sizeOf(other.type); // ignore padding if any
    bytes.insert(bytes.end(), other.bytes.begin(), other.bytes.begin() + addbytesWithoutPadding);

    // grow to align to 256 bytes boundary (will be undone when more pieces are appended)
    size_t multiplier = (size_t)ceil((float)bytes.size() / (float)256);
    bytes.resize(multiplier * 256);
  }

  template <typename From, typename To>
  void convertFromTo() {
    size_t elements = size() / sizeof(From);
    size_t newSize = elements * sizeof(To);
    std::vector<char> newBytes(newSize);

    From* in = (From*)bytes.data();
    To* out = (To*)newBytes.data();
    for(int i = 0; i < elements; ++i)
      out[i] = (To)in[i];

    bytes.swap(newBytes);
  }

  template <typename T>
  void convertTo() {
    if(type == Type::float32)
      convertFromTo<float, T>();
    else if(type == Type::float16)
      convertFromTo<HalfFloat, T>();
    else 
      ABORT("convert from type {} not implemented", type);
  }

  void convert(Type toType) {
    if(type == toType)
      return;

    if(toType == Type::float32)
      convertTo<float>();
    else if(toType == Type::float16)
      convertTo<float16>();
    else
      ABORT("convert to type {} not implemented", toType);

    type = toType;
  }
};

}  // namespace io
}  // namespace marian
