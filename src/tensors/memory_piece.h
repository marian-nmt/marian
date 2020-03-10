#pragma once

#include "common/definitions.h"

#include <iostream>

namespace marian {

class MemoryPiece {
private:
  uint8_t* data_;
  size_t size_;

  ENABLE_INTRUSIVE_PTR(MemoryPiece)
  
  // Contructor is private, use MemoryPiece::New(...)
  MemoryPiece(uint8_t* data, size_t size) : data_(data), size_(size) {}

public:
  // Use this whenever pointing to MemoryPiece
  typedef IPtr<MemoryPiece> PtrType;

  // Use this whenever creating a pointer to MemoryPiece
  template <class ...Args>
  static PtrType New(Args&& ...args) {
    return PtrType(new MemoryPiece(std::forward<Args>(args)...));
  }
  
  uint8_t* data() const { return data_; }
  uint8_t* data() { return data_; }

  template <typename T>
  T* data() const {
    return (T*)data_;
  }

  template <typename T>
  T* data() {
    return (T*)data_;
  }

  size_t size() const { return size_; }

  void set(uint8_t* data, size_t size) {
    data_ = data;
    size_ = size;
  }

  void setPtr(uint8_t* data) { data_ = data; }

  friend std::ostream& operator<<(std::ostream& out, const MemoryPiece mp) {
    out << "MemoryPiece - ptr: " << std::hex << (size_t)mp.data() << std::dec
        << " size: " << mp.size();
    return out;
  }

};
}  // namespace marian
