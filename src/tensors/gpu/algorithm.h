#pragma once

#include "tensors/backend.h"

namespace marian {
  namespace gpu {
    template <typename T>
    void copy(Ptr<Backend> backend, const T* begin, const T* end, T* dest);
    
    void fill(Ptr<Backend> backend, float* begin, float* end, float value);

    void setSparse(Ptr<Backend> backend, const std::vector<size_t>&, const std::vector<float>&, float*);
  }
}
