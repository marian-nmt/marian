#pragma once

#include "tensors/tensor.h"

namespace marian {

namespace gpu {

template <class Functor, class... Tensors>
void Add(Functor functor, float scale, marian::Tensor out, Tensors... tensors);

template <class Functor, class AggFunctor, class... Tensors>
void Aggregate(Functor functor, float initAgg, AggFunctor aggFunctor, float scale, marian::Tensor out, Tensors... tensors);
}
}  // namespace marian
