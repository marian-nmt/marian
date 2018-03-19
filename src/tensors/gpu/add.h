#pragma once

#include "tensors/tensor.h"

namespace marian {

namespace gpu {

template <class Functor, class... Tensors>
void Add(Functor functor, float scale, marian::Tensor out, Tensors... tensors);
}
}
