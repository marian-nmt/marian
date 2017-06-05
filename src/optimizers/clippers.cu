#include "clippers.h"

#include "kernels/tensor_operators.h"
#include "kernels/thrust_functions.h"

namespace marian {
void Elementwise::clip(Tensor t) {
  Element(_1 = Clip(_1, c_), t);
}

void Norm::clip(Tensor t) {
  float l2Norm = L2Norm(t);
  if(l2Norm >= c_)
    Element(_1 = (c_ / l2Norm) * _1, t);
}
}