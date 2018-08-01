#pragma once

#include "tensors/tensor.h"

namespace marian {
namespace gpu {

template <class Functor, class... Tensors>
void Element(Functor functor, Tensor out, Tensors... tensors);
}
}  // namespace marian
