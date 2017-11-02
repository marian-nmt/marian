#include <iostream>

#include "gpu/placeholders.h"
#include "gpu/functions.h"

int main(int argc, char** argv) {

  using namespace marian::functional;

  ref<1> x;
  ref<2> y;
  auto clip = if_then_else(abs(x) > 1, sgn(x), x);

  std::cerr << clip(2.f) << " " << sizeof(clip) << std::endl;
  std::cerr << clip(-2.f) << " " << sizeof(clip) << std::endl;
  std::cerr << clip(-0.2f) << " " << sizeof(clip) << std::endl;

  return 0;
}

/*

struct SwishNodeOp : public UnaryNodeOp {
  template <typename... Args>
  SwishNodeOp(Args... args) : UnaryNodeOp(args...) {}

  NodeOps forwardOps() {

    using namespace gpu::m;
    ref<1> x;
    auto swish = x * logit(x);

    return {NodeOp(Element(swish, val_, child(0)->val()))};
  }

  NodeOps backwardOps() {

    using namespace gpu::m;
    ref<0> dJdf;
    ref<1> x;
    ref<2> f;
    auto dJdx = dJdf * (f + logit(x) * (1 - f));

    return {NodeOp(Add(dJdx, child(0)->grad(), adj_, child(0)->val(), val_))};
  }

 */