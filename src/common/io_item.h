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

  size_t size() const {
    if(mapped)
      return shape.elements() * sizeOf(type);
    else
      return bytes.size();
  }
};

}  // namespace io
}  // namespace marian
