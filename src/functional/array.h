#pragma once

#include "functional/defs.h"

namespace marian {

namespace functional {

template <typename T, size_t N>
struct Array {
  typedef T value_type;
  T data_[N];

  HOST_DEVICE_INLINE const T* data() const { return data_; }

  HOST_DEVICE_INLINE T* data() { return data_; }

  HOST_DEVICE_INLINE constexpr static size_t size() { return N; }

  HOST_DEVICE_INLINE T& operator[](size_t i) { return data_[i]; }
  HOST_DEVICE_INLINE const T& operator[](size_t i) const { return data_[i]; }

  HOST_DEVICE_INLINE T* begin() { return data_; }
  HOST_DEVICE_INLINE const T* begin() const { return data_; }

  HOST_DEVICE_INLINE T* end() { return data_ + N; }
  HOST_DEVICE_INLINE const T* end() const { return data_ + N; }

  HOST_DEVICE_INLINE void fill(T val) {
    for(int i = 0; i < N; ++i)
      data_[i] = val;
  }

  HOST_DEVICE_INLINE T& back() { return data_[N - 1]; }
  HOST_DEVICE_INLINE const T& back() const { return data_[N - 1]; }

  HOST_DEVICE_INLINE bool operator==(const Array<T, N>& other) {
    for(int i = 0; i < N; ++i)
      if(data_[i] != other[i])
        return false;
    return true;
  }
};
}  // namespace functional
}  // namespace marian
