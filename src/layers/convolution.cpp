#include "layers/convolution.h"
#include "graph/node_operators_binary.h"

namespace marian {

#ifdef CUDNN
Convolution::Convolution(Ptr<ExpressionGraph> graph) {}

Expr Convolution::apply(Expr x) {
  auto graph = x->graph();

  auto prefix = opt<std::string>("prefix");
  auto kernelDims = opt<std::pair<int, int>>("kernel-dims");
  auto kernelNum = opt<int>("kernel-num");
  auto paddings = opt<std::pair<int, int>>("paddings", std::make_pair(0, 0));
  auto strides = opt<std::pair<int, int>>("strides", std::make_pair(1, 1));

  int layerIn = x->shape()[1];
  auto kernel
      = graph->param(prefix + "_conv_kernels",
                      {layerIn, kernelNum, kernelDims.first, kernelDims.second},
                      inits::glorotUniform());

  auto bias = graph->param(
      prefix + "_conv_bias", {1, kernelNum, 1, 1}, inits::zeros());

  std::vector<Expr> nodes = {x, kernel, bias};
  return Expression<ConvolutionOp>(
      nodes, paddings.first, paddings.second, strides.first, strides.second);
}

Expr Convolution::apply(const std::vector<Expr>&) {
  ABORT("Can't apply convolution on many inputs at once");
  return nullptr;
}
#endif

}  // namespace marian
