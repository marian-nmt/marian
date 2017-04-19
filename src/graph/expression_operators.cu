#include "graph/expression_operators.h"
#include "graph/node_operators.h"
#include "graph/node_operators_unary.h"

namespace marian {

Expr debug(Expr a, const std::string& message) {
  a->debug(message);
  return a;
}

Expr name(Expr a, const std::string& name) {
  a->set_name(name);
  a->graph()->add_named_node(a, name);
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

//Expr batch_norm(Expr x, Expr gamma, Expr beta) {
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

Expr lexical_bias(Expr logits, Expr att, Ptr<sparse::CSR> lf) {
  return Expression<LexicalProbNodeOp>(logits, att, lf);
}

}
