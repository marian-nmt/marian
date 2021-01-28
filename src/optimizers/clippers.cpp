#include "clippers.h"

#include "functional/functional.h"
#include "tensors/tensor_operators.h"

namespace marian {
float ElementwiseClipper::clip(Tensor t, float costScalingFactor) {
  using namespace functional;
  Element(_1 = functional::clip(_1, c_ * costScalingFactor), t);
  return 0.f; // dummy
}

float NormClipper::clip(Tensor t, float costScalingFactor) {
  using namespace functional;
  float l2Norm = L2Norm(t, allocator_);
  float clipValue = c_ * costScalingFactor;
  if(l2Norm > clipValue) {
    LOG(debug, "Re-scaling gradient by {}", clipValue / l2Norm);
    Element(_1 = (clipValue / l2Norm) * _1, t);
  }
  return l2Norm;
}

// don't clip, just report L2Norm
float ReportNormClipper::clip(Tensor t, float /*costScalingFactor*/) {
  using namespace functional;
  return L2Norm(t, allocator_);
}

}  // namespace marian
