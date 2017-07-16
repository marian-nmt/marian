#include "graph/expression_operators.h"
#include "kernels/sparse.h"

#include "graph/node_operators.h"
#include "graph/node_operators_binary.h"
#include "graph/node_operators_unary.h"

namespace marian {

Expr debug(Expr a, const std::string& message) {
  a->debug(message);
  return a;
}

Expr rows(Expr a, const std::vector<size_t>& indeces) {
  return Expression<RowsNodeOp>(a, indeces);
}

Expr cols(Expr a, const std::vector<size_t>& indeces) {
  return Expression<ColsNodeOp>(a, indeces);
}

Expr logit(Expr a) {
  return Expression<LogitNodeOp>(a);
}

Expr relu(Expr a) {
  return Expression<ReLUNodeOp>(a);
}

Expr log(Expr a) {
  return Expression<LogNodeOp>(a);
};

Expr exp(Expr a) {
  return Expression<ExpNodeOp>(a);
};

Expr operator-(Expr a) {
  return Expression<NegNodeOp>(a);
};

Expr softmax(Expr a, Expr mask) {
  return Expression<SoftmaxNodeOp>(a, mask);
}

Expr logsoftmax(Expr a) {
  return Expression<LogSoftmaxNodeOp>(a);
}

/*********************************************************/

Expr operator+(Expr a, Expr b) {
  return Expression<PlusNodeOp>(a, b);
}

Expr operator-(Expr a, Expr b) {
  return Expression<MinusNodeOp>(a, b);
}

Expr operator*(Expr a, Expr b) {
  return Expression<MultNodeOp>(a, b);
}

Expr operator/(Expr a, Expr b) {
  return Expression<DivNodeOp>(a, b);
}

/*********************************************************/

Expr operator+(Expr a, float b) {
  return Expression<ScalarAddNodeOp>(a, b);
}

Expr operator+(float a, Expr b) {
  return Expression<ScalarAddNodeOp>(b, a);
}

Expr operator-(Expr a, float b) {
  return Expression<ScalarAddNodeOp>(a, -b);
}

Expr operator-(float a, Expr b) {
  return Expression<ScalarAddNodeOp>(-b, a);
}

Expr operator*(float a, Expr b) {
  return Expression<ScalarMultNodeOp>(b, a);
}

Expr operator*(Expr a, float b) {
  return Expression<ScalarMultNodeOp>(a, b);
}

Expr operator/(Expr a, float b) {
  return Expression<ScalarMultNodeOp>(a, 1.f / b);
}

/*********************************************************/

Expr concatenate(const std::vector<Expr>& concats, keywords::axis_k ax) {
  return Expression<ConcatenateNodeOp>(concats, ax);
}

Expr reshape(Expr a, Shape shape) {
  return Expression<ReshapeNodeOp>(a, shape);
}

Expr flatten(Expr a) {
  Shape shape = {a->shape().elements()};
  return Expression<ReshapeNodeOp>(a, shape);
}

Expr sum(Expr a, keywords::axis_k ax) {
  return Expression<SumNodeOp>(a, ax);
}

Expr mean(Expr a, keywords::axis_k ax) {
  return Expression<MeanNodeOp>(a, ax);
}

Expr scalar_product(Expr a, Expr b, keywords::axis_k ax) {
  return Expression<ScalarProductNodeOp>(a, b, ax);
}

Expr weighted_average(Expr in, Expr weights, keywords::axis_k ax) {
  auto p = scalar_product(in, weights, ax);
  auto s = sum(weights, ax);
  return p / s;
}

Expr dot(Expr a, Expr b) {
  return Expression<DotNodeOp>(a, b);
}

Expr transpose(Expr a) {
  return Expression<TransposeNodeOp>(a);
}

Expr step(Expr a, size_t step) {
  return Expression<TimestepNodeOp>(a, step);
}

Expr cross_entropy(Expr a, Expr b) {
  auto sOrig = a->shape();
  auto sOut = a->shape();
  Shape sTemp({sOrig[0] * sOrig[2] * sOrig[3], sOrig[1], 1, 1});
  sOut.set(1, 1);
  return reshape(Expression<CrossEntropyNodeOp>(reshape(a, sTemp), b), sOut);

  //return Expression<CrossEntropyNodeOp>(a, b);
}

Expr affine(Expr a, Expr b, Expr c) {
  std::vector<Expr> nodes = {a, b, c};
  return Expression<AffineNodeOp>(nodes);
}

Expr plus(const std::vector<Expr>&) {
  UTIL_THROW2("Not implemented");
}

Expr tanh(const std::vector<Expr>& nodes) {
  return Expression<TanhNodeOp>(nodes);
}

Expr logit(const std::vector<Expr>&) {
  UTIL_THROW2("Not implemented");
}

Expr relu(const std::vector<Expr>&) {
  UTIL_THROW2("Not implemented");
}

Expr sqrt(Expr a, float eps) {
  return Expression<SqrtNodeOp>(a, eps);
}

Expr square(Expr a) {
  return Expression<SquareNodeOp>(a);
}

Expr layer_norm(Expr x, Expr gamma, Expr beta) {
  std::vector<Expr> nodes = {x, gamma};
  if(beta)
    nodes.push_back(beta);
  return Expression<LayerNormalizationOp>(nodes);
}

// Expr batch_norm(Expr x, Expr gamma, Expr beta) {
//  auto mju = mean(x, keywords::axis=0);
//  auto xmmju = x - mju;
//  auto std = sqrt(mean(square(xmmju), keywords::axis=0), 1e-9);
//
//  if(beta)
//    return gamma * (xmmju / std) + beta;
//  else
//    return gamma * (xmmju / std);
//}

Expr shift(Expr a, Shape shift) {
  return Expression<ShiftNodeOp>(a, shift);
}

Expr lexical_bias(Expr logits, Expr att, float eps, Ptr<sparse::CSR> lf) {
  return Expression<LexicalProbNodeOp>(logits, att, eps, lf);
}

#ifdef CUDNN

Expr convolution(Expr x, Expr filters, Expr bias) {
  std::vector<Expr> nodes = {x, filters, bias};
  return Expression<ConvolutionOp>(nodes);
}

Expr avg_pooling(
    Expr x,
    int height, int width,
    int padHeight, int padWidth,
    int strideHeight, int strideWidth)
{
  return Expression<PoolingOp>(x,
      height, width,
      padHeight, padWidth,
      strideHeight, strideWidth,
      PoolingOp::Mode::AVERAGE_POOLING);
}

Expr max_pooling(
    Expr x,
    int height, int width,
    int padHeight, int padWidth,
    int strideHeight, int strideWidth)
{
  return Expression<PoolingOp>(x,
      height, width,
      padHeight, padWidth,
      strideHeight, strideWidth,
      PoolingOp::Mode::MAX_POOLING);
}

#endif

}
