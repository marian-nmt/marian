#pragma once

#include "tensors/backend.h"

namespace marian {
namespace gpu {
template <typename T>
void copy(Ptr<marian::Backend> backend, const T* begin, const T* end, T* dest);

template <typename T>
void fill(Ptr<marian::Backend> backend, T* begin, T* end, T value);

void setSparse(Ptr<marian::Backend> backend,
               const std::vector<size_t>&,
               const std::vector<float>&,
               float*);
}  // namespace gpu
}  // namespace marian
