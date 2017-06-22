#pragma once
#include "graph/expression_graph.h"

namespace marian {

Expr debug(Expr a, const std::string& message = "");

Expr rows(Expr a, const std::vector<size_t>& indeces);
Expr cols(Expr a, const std::vector<size_t>& indeces);

Expr plus(const std::vector<Expr>&);

Expr logit(Expr a);
Expr logit(const std::vector<Expr>&);

Expr tanh(const std::vector<Expr>&);

template <typename... Args>
Expr tanh(Args... args) {
  std::vector<Expr> nodes{args...};
  return tanh(nodes);
}

Expr relu(Expr a);
Expr relu(const std::vector<Expr>&);

Expr log(Expr a);

Expr exp(Expr a);

Expr operator-(Expr a);

/*********************************************************/

Expr operator+(Expr a, Expr b);
Expr operator+(float a, Expr b);
Expr operator+(Expr a, float b);

Expr operator-(Expr a, Expr b);
Expr operator-(float a, Expr b);
Expr operator-(Expr a, float b);

Expr operator*(Expr a, Expr b);
Expr operator*(float a, Expr b);
Expr operator*(Expr a, float b);

Expr operator/(Expr a, Expr b);
Expr operator/(float a, Expr b);
Expr operator/(Expr a, float b);

Expr dot(Expr a, Expr b);

Expr transpose(Expr a);

Expr concatenate(const std::vector<Expr>& concats, keywords::axis_k ax = 0);

Expr reshape(Expr a, Shape shape);

Expr flatten(Expr a);

/*********************************************************/

Expr sum(Expr a, keywords::axis_k ax = 0);

Expr softmax(Expr a, Expr mask = nullptr);

Expr logsoftmax(Expr a);

Expr mean(Expr a, keywords::axis_k ax = 0);

Expr cross_entropy(Expr a, Expr b);

Expr affine(Expr a, Expr b, Expr c);

Expr scalar_product(Expr a, Expr b, keywords::axis_k ax = 0);

Expr weighted_average(Expr in, Expr weights, keywords::axis_k ax = 0);

Expr step(Expr a, size_t step);

Expr sqrt(Expr a, float eps = 0.f);
Expr square(Expr a);

Expr layer_norm(Expr x, Expr gamma, Expr beta = nullptr);

template <typename... Args>
Expr dropout(Expr x, Args... args) {
  auto mask = Get(keywords::mask, nullptr, args...);
  float dropout_prob = Get(keywords::dropout_prob, 0.0f, args...);

  UTIL_THROW_IF2(!mask && !dropout_prob, "Neither mask nor dropout prob given");
  if(!mask) {
    auto graph = x->graph();
    mask = graph->dropout(dropout_prob, x->shape());
  }
  return x * mask;
}

Expr shift(Expr, Shape);

Expr convolution(Expr x, Expr filters, Expr bias);

Expr avg_pooling(
        Expr x,
        int height, int width,
        int padHeight, int padWidth,
        int strideHeight, int strideWidth);

Expr max_pooling(
        Expr x,
        int height, int width,
        int padHeight, int padWidth,
        int strideHeight, int strideWidth);
}
