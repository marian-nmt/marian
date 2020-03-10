#include "clippers.h"

#include "functional/functional.h"
#include "tensors/tensor_operators.h"

namespace marian {
void Elementwise::clip(Tensor t) {
  using namespace functional;
  Element(_1 = functional::clip(_1, c_), t);
}

void Norm::clip(Tensor t) {
  using namespace functional;
  float l2Norm = L2Norm(t, nullptr); // @TODO: this is a placeholder for a memory allocator, will be replaced with better version in a PR or two.
  if(l2Norm >= c_)
    Element(_1 = (c_ / l2Norm) * _1, t);
}
}  // namespace marian
