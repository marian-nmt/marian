#pragma once

#include <map>
#include <memory>

#include "tensors/tensor.h"
#include "tensors/allocator.h"

namespace marian {

class Clipper {
protected:
  Ptr<Allocator> allocator_;

public:
  virtual ~Clipper() {}

  virtual float clip(Tensor, float /*costScalingFactor*/ = 1.f) = 0;
  virtual void setAllocator(Ptr<Allocator> allocator) { allocator_ = allocator; }
};

class ElementwiseClipper : public Clipper {
public:
  ElementwiseClipper(float c = 10.0) : c_(c) {}
  ~ElementwiseClipper() override {}

  float clip(Tensor t, float costScalingFactor = 1.f) override;

private:
  float c_;
};

class NormClipper : public Clipper {
public:
  NormClipper(float c = 1.0) : c_(c) {}
  ~NormClipper() override {}

  float clip(Tensor t, float costScalingFactor = 1.f) override;

private:
  float c_;
};

// don't clip, just report Froebenius norm
class ReportNormClipper : public Clipper {
public:
  ReportNormClipper(float /*c = 1.0*/)  {}
  ~ReportNormClipper() override {}

  float clip(Tensor t, float costScalingFactor = 1.f) override;
};

}  // namespace marian
