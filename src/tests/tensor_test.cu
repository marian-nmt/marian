#include <iostream>

#include "gpu/placeholders.h"
#include "gpu/functions.h"

int main(int argc, char** argv) {

  using namespace marian::functional;

  ref<1> x;
  ref<2> y;
  auto func = x = if_then_else(x > y, x + y, y);

  float a = 1;
  float b = 1;
  std::cerr << func(a, 0.1f) << " " << sizeof(func) << std::endl;
  std::cerr << func(b, 1.2f) << " " << sizeof(func) << std::endl;

  std::cerr << a << " " << b << std::endl;

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