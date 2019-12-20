#pragma once
#include "graph/expression_graph.h"
#include "graph/node_initializers.h"

namespace marian {

Expr debug(Expr a, const std::string& message = "");

Expr checkpoint(Expr a);

typedef Expr(ActivationFunction)(Expr);

Expr plus(const std::vector<Expr>&);

// TODO: should be logistic(), not sigmoid()
Expr sigmoid(Expr a);
Expr sigmoid(const std::vector<Expr>&);

Expr swish(Expr a);
Expr swish(const std::vector<Expr>&);

Expr gelu(Expr a);
Expr gelu(const std::vector<Expr>&);

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

// Note: Following numpy, minimum() is element-wise, while min() is along an axis in both Numpy and PyTorch.
Expr maximum(Expr a, Expr b);
Expr minimum(Expr a, Expr b);

// Note: We cannot overload the relational operators, as they also mean something for Expr itself.
// Note: These names follow PyTorch convention.
Expr lt(Expr a, Expr b);
Expr eq(Expr a, Expr b);
Expr gt(Expr a, Expr b);
Expr ge(Expr a, Expr b);
Expr ne(Expr a, Expr b);
Expr le(Expr a, Expr b);

Expr lt(float a, Expr b);
Expr eq(float a, Expr b);
Expr gt(float a, Expr b);
Expr ge(float a, Expr b);
Expr ne(float a, Expr b);
Expr le(float a, Expr b);

Expr lt(Expr a, float b);
Expr eq(Expr a, float b);
Expr gt(Expr a, float b);
Expr ge(Expr a, float b);
Expr ne(Expr a, float b);
Expr le(Expr a, float b);

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

Expr csr_dot(const Shape& A_shape, Expr Avalues, Expr Aindices, Expr Aoffsets, Expr B, bool transA = false);
Expr dot_csr(Expr A, const Shape& B_shape, Expr B_values, Expr B_indices, Expr B_offsets, bool transB = false);

Expr transpose(Expr a);
Expr transpose(Expr a, const std::vector<int>& axes);

Expr swapAxes(Expr x, int axis1, int axis2);

Expr cast(Expr a, Type type = Type::float32);

Expr concatenate(const std::vector<Expr>& concats, int ax = 0);
Expr repeat(Expr a, size_t repeats, int ax = 0);

Expr reshape(Expr a, Shape shape);

Expr clipGradient(Expr a, float clipValue);

Expr atleast_1d(Expr a);
Expr atleast_2d(Expr a);
Expr atleast_3d(Expr a);
Expr atleast_4d(Expr a);
Expr atleast_nd(Expr a, size_t dims);

// create a constant of shape a->shape() and initialize with init
// @TODO: add a && version, to avoid a ref count. NodeInitializers are typically temps.
// @TODO: and/or make this a template on init
static inline Expr constant_like(Expr a, const Ptr<inits::NodeInitializer>& init) {
  return a->graph()->constant(a->shape(), init, a->value_type());
}

// short-cut to init from std::vector, since we do this so often
template<typename ElementType>
Expr constant_like(Expr a, const std::vector<ElementType>& v) { return constant_like(a, inits::fromVector(std::move(v))); }
template<typename ElementType>
Expr constant_like(Expr a, std::vector<ElementType>&& v) { return constant_like(a, inits::fromVector(v)); }

Expr flatten(Expr a);
Expr flatten_2d(Expr a);

Expr stopGradient(Expr a);

Expr gather(Expr a, int axis, Expr indices);

// Warning: Don't try to pass a scalar literal 0 as indices; it will compile but pass nullptr...
Expr index_select(Expr a, int axis, Expr indices);

// convenience wrappers for index_select()
Expr index_select(Expr a, int axis, const std::vector<IndexType>& indices);
static inline Expr rows(Expr a, Expr indices) {
  return index_select(a, 0, indices);
}
static inline Expr rows(Expr a, const std::vector<IndexType>& indexVector) {
  return index_select(a, 0, indexVector);
}
static inline Expr cols(Expr a, Expr indices) {
  return index_select(a, -1, indices);
}
static inline Expr cols(Expr a, const std::vector<IndexType>& indexVector) {
  return index_select(a, -1, indexVector);
}

Expr slice(Expr a, int axis, Slice slice);

// convenience wrappers for slice()
static inline Expr slice(Expr a, int axis, int index) { // single index  @NOTE: This was formerlly called step()
  return slice(a, axis, Slice(index));
}

static inline Expr narrow(Expr a, int axis, size_t start, size_t length) { // PyTorch name
  return slice(a, axis, Slice((int)start, (int)(start + length)));
}

/*********************************************************/

Expr sum(Expr a, int ax = 0);
Expr mean(Expr a, int ax = 0);
Expr std(Expr a, int ax);
Expr var(Expr a, int ax);
Expr max(Expr a, int ax);
Expr min(Expr a, int ax);
Expr prod(Expr a, int ax);
Expr logsumexp(Expr a, int ax);

Expr softmax(Expr x, int axis = -1);

// @TODO: maybe get rid of this entirely to not obfuscate, what's going on inside.
// @TODO: switch to log-masking everywhere?
Expr softmax(Expr a, Expr zeroOneMask, int axis = -1);

Expr logsoftmax(Expr a);

Expr cross_entropy(Expr a, Expr b);

Expr unlikelihood(Expr a, Expr b);

Expr scalar_product(Expr a, Expr b, int ax = 0);

Expr weighted_average(Expr in, Expr weights, int ax = 0);

Expr sqrt(Expr a, float eps = 0.f);
Expr square(Expr a);

Expr layerNorm(Expr x, Expr gamma, Expr beta = nullptr, float eps = 1e-9);

Expr highway(Expr y, Expr x, Expr t);
Expr highway(const std::string prefix, Expr x);

static inline Expr dropout(Expr x, Expr mask) {
  if (mask)
    return x * mask;
  else
    return x;
}

static inline Expr dropout(Expr x, float dropProb, Shape shape) {
  if(dropProb == 0)
    return x;
  auto graph = x->graph();
  auto mask = graph->dropoutMask(dropProb, shape);
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
