#include <iostream>

#include "gpu/placeholders.h"
#include "gpu/functions.h"

int main(int argc, char** argv) {

  using namespace marian::functional;

  auto func = _1 = tanh(_2) * 3;

  float z;
  std::cerr << func(z, 2.f) << " " << sizeof(func) << std::endl;
  std::cerr << z << " " << std::endl;
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