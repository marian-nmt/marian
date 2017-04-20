#pragma once
#include "graph/expression_graph.h"
#include "kernels/sparse.h"

namespace marian {

Expr training(Expr a);

Expr inference(Expr a);

Expr debug(Expr a, const std::string& message = "");

Expr name(Expr a, const std::string& name);

Expr rows(Expr a, const std::vector<size_t>& indeces);
Expr cols(Expr a, const std::vector<size_t>& indeces);

Expr plus(const std::vector<Expr>&);

Expr logit(Expr a);
Expr logit(const std::vector<Expr>&);

Expr tanh(const std::vector<Expr>&);

template <typename ...Args>
Expr tanh(Args ...args) {
  std::vector<Expr> nodes{args...};
  return Expression<TanhNodeOp>(nodes);
}

Expr relu(Expr a);
Expr relu(const std::vector<Expr>&);

Expr log(Expr a);

Expr exp(Expr a);

Expr operator-(Expr a);

/*********************************************************/

Expr operator+(Expr a, Expr b);

Expr operator-(Expr a, Expr b);

Expr operator*(Expr a, Expr b);

Expr operator/(Expr a, Expr b);

Expr dot(Expr a, Expr b);

Expr transpose(Expr a);

template <typename ...Args>
Expr concatenate(const std::vector<Expr>& concats, Args ...args) {
  return Expression<ConcatenateNodeOp>(concats, args...);
}

template <typename ...Args>
Expr reshape(Expr a, Shape shape, Args ...args) {
  return Expression<ReshapeNodeOp>(a, shape, args...);
}

/*********************************************************/

template <typename ...Args>
Expr sum(Expr a, Args ...args) {
  return Expression<SumNodeOp>(a, args...);
}

Expr softmax(Expr a, Expr mask = nullptr);

Expr logsoftmax(Expr a);

template <typename ...Args>
Expr mean(Expr a, Args ...args) {
  return Expression<MeanNodeOp>(a, args...);
}

Expr cross_entropy(Expr a, Expr b);

Expr affine(Expr a, Expr b, Expr c);

template <typename ...Args>
Expr scalar_product(Expr a, Expr b, Args ...args) {
  return Expression<ScalarProductNodeOp>(a, b, args...);
}

template <typename ...Args>
Expr weighted_average(Expr in, Expr weights, Args ...args) {
  auto p = scalar_product(in, weights, args...);
  auto s = sum(weights, args...);
  return p / s;
}

Expr step(Expr a, size_t step);

Expr sqrt(Expr a, float eps = 0.f);
Expr square(Expr a);

Expr layer_norm(Expr x, Expr gamma, Expr beta = nullptr);

template <typename ...Args>
Expr dropout(Expr x, Args ...args) {
  auto mask = Get(keywords::mask, nullptr, args...);
  float dropout_prob = Get(keywords::dropout_prob, 0.0f, args...);

  UTIL_THROW_IF2(!mask && !dropout_prob,
                 "Neither mask nor dropout prob given");
  if(!mask) {
    auto graph = x->graph();
    mask = graph->dropout(dropout_prob, x->shape());
  }
  return x * mask;
}

Expr shift(Expr, Shape);

Expr lexical_bias(Expr logits, Expr att, Expr exp, Ptr<sparse::CSR> lf);
  
}
