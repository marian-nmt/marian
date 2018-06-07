#pragma once
#include "graph/expression_graph.h"

namespace marian {

Expr debug(Expr a, const std::string& message = "");

Expr plus(const std::vector<Expr>&);

Expr logit(Expr a);
Expr logit(const std::vector<Expr>&);

Expr swish(Expr a);
Expr swish(const std::vector<Expr>&);

Expr tanh(const std::vector<Expr>&);

template <typename... Args>
Expr tanh(Args... args) {
  std::vector<Expr> nodes{args...};
  return tanh(nodes);
}

Expr relu(Expr a);
Expr relu(const std::vector<Expr>&);

Expr leakyrelu(Expr a);
Expr leakyrelu(const std::vector<Expr>&);

Expr prelu(Expr a, float alpha = 0.01);
Expr prelu(const std::vector<Expr>&, float alpha = 0.01);

Expr log(Expr a);

Expr exp(Expr a);

Expr clip(Expr a, float c);

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

// Expr pow(Expr a, Expr b);
// Expr pow(float a, Expr b);
// Expr pow(Expr a, float b);

Expr dot(Expr a,
         Expr b,
         bool transA = false,
         bool transB = false,
         float scalar = 1.f);

Expr bdot(Expr a,
          Expr b,
          bool transA = false,
          bool transB = false,
          float scalar = 1.f);

Expr affine(Expr a,
            Expr b,
            Expr c,
            bool transA = false,
            bool transB = false,
            float scalar = 1.f);

Expr transpose(Expr a);
Expr transpose(Expr a, const std::vector<int>& axes);

Expr concatenate(const std::vector<Expr>& concats, keywords::axis_k ax = 0);
Expr repeat(Expr a, size_t repeats, keywords::axis_k ax = 0);

Expr reshape(Expr a, Shape shape);

Expr atleast_1d(Expr a);
Expr atleast_2d(Expr a);
Expr atleast_3d(Expr a);
Expr atleast_4d(Expr a);
Expr atleast_nd(Expr a, size_t dims);

Expr flatten(Expr a);
Expr flatten_2d(Expr a);

Expr rows(Expr a, const std::vector<size_t>& indices);
Expr cols(Expr a, const std::vector<size_t>& indices);

Expr select(Expr a, int axis, const std::vector<size_t>& indices);

/*********************************************************/

Expr sum(Expr a, keywords::axis_k ax = 0);

Expr softmax(Expr a, Expr mask = nullptr);

Expr logsoftmax(Expr a);

Expr mean(Expr a, keywords::axis_k ax = 0);

Expr cross_entropy(Expr a, Expr b);

Expr scalar_product(Expr a, Expr b, keywords::axis_k ax = 0);

Expr weighted_average(Expr in, Expr weights, keywords::axis_k ax = 0);

Expr step(Expr a, int step, int axis);

Expr sqrt(Expr a, float eps = 0.f);
Expr square(Expr a);

Expr layer_norm(Expr x, Expr gamma, Expr beta = nullptr, float eps = 1e-9);

Expr highway(Expr y, Expr x, Expr t);
Expr highway(const std::string prefix, Expr x);

static inline Expr dropout(Expr x, Expr mask) {
  return x * mask;
}

static inline Expr dropout(Expr x, float prob, Shape shape) {
  auto graph = x->graph();
  auto mask = graph->dropout(prob, shape);
  return dropout(x, mask);
}

static inline Expr dropout(Expr x, float prob) {
  return dropout(x, prob, x->shape());
}

Expr shift(Expr, Shape);

Expr convert2cudnnFormat(Expr x);

Expr convertFromcudnnFormat(Expr x);

Expr avg_pooling(Expr x,
                 int height,
                 int width,
                 int padHeight = 0,
                 int padWidth = 0,
                 int strideHeight = 1,
                 int strideWidth = 1);

Expr max_pooling(Expr x,
                 int height,
                 int width,
                 int padHeight = 0,
                 int padWidth = 0,
                 int strideHeight = 1,
                 int strideWidth = 1);

Expr pooling_with_masking(Expr x, Expr mask, int width, bool isEven = false);
}
