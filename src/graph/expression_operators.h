#pragma once
#include "graph/expression_graph.h"
#include "graph/node_initializers.h"

namespace marian {

Expr debug(Expr a, const std::string& message = "");

typedef Expr(ActivationFunction)(Expr);

Expr plus(const std::vector<Expr>&);

// TODO: should be logistic(), not sigmoid()
Expr sigmoid(Expr a);
Expr sigmoid(const std::vector<Expr>&);

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

Expr logaddexp(Expr a, Expr b);

Expr max(Expr a, Expr b);  // TODO: haggle over the name (max vs. elementMax)

Expr min(Expr a, Expr b);  // TODO: haggle over the name

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

Expr swapAxes(Expr x, int axis1, int axis2);

Expr concatenate(const std::vector<Expr>& concats, int ax = 0);
Expr repeat(Expr a, size_t repeats, int ax = 0);

Expr reshape(Expr a, Shape shape);

Expr atleast_1d(Expr a);
Expr atleast_2d(Expr a);
Expr atleast_3d(Expr a);
Expr atleast_4d(Expr a);
Expr atleast_nd(Expr a, size_t dims);

// create a constant of shape a->shape() and initialize with init
Expr constant_like(Expr a, const NodeInitializer& init);

Expr flatten(Expr a);
Expr flatten_2d(Expr a);

Expr rows(Expr a, Expr indices);
Expr rows(Expr a, const std::vector<IndexType>& indices);

Expr cols(Expr a, Expr indices);
Expr cols(Expr a, const std::vector<IndexType>& indices);

Expr select(Expr a, Expr indices, int axis);
Expr select(Expr a, const std::vector<IndexType>& indices, int axis);

/*********************************************************/

Expr sum(Expr a, int ax = 0);

Expr softmax(Expr x, int axis = -1);

// @TODO: maybe get rid of this entirely to not obfuscate, what's going on inside.
// @TODO: switch to log-masking everywhere?
Expr softmax(Expr a, Expr zeroOneMask, int axis = -1);

Expr logsoftmax(Expr a);

Expr mean(Expr a, int ax = 0);

Expr cross_entropy(Expr a, Expr b);

Expr scalar_product(Expr a, Expr b, int ax = 0);

Expr weighted_average(Expr in, Expr weights, int ax = 0);

Expr step(Expr a, int step, int axis);

Expr sqrt(Expr a, float eps = 0.f);
Expr square(Expr a);

Expr layerNorm(Expr x, Expr gamma, Expr beta = nullptr, float eps = 1e-9);

Expr highway(Expr y, Expr x, Expr t);
Expr highway(const std::string prefix, Expr x);

static inline Expr dropout(Expr x, Expr mask) {
  return x * mask;
}

static inline Expr dropout(Expr x, float dropProb, Shape shape) {
  if(dropProb == 0)
    return x;
  auto graph = x->graph();
  auto mask = graph->dropout(dropProb, shape);
  return dropout(x, mask);
}

static inline Expr dropout(Expr x, float dropProb) {
  if(dropProb == 0)
    return x;
  return dropout(x, dropProb, x->shape());
}

Expr shift(Expr, Shape, float padValue = 0);

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
}  // namespace marian
