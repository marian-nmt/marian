// TODO: move to backend, into graph/
#pragma once

#include "common/config.h"
#include "tensors/tensor.h"
#include "tensors/tensor_operators.h"

#include <functional>
#include <random>

namespace marian {

class ExpressionGraph; // Forward declaration

namespace inits {

/**
 * Base class for specialized NodeInitializers.
 *
 * A NodeInitializer is a functor that can be invoked on a tensor
 * during node intialization. You need to override operator()(Tensor)
 * with your own functionality.
 *
 * See node_initializers.cpp for examples.
 */
class NodeInitializer {
protected:
  Weak<Allocator> allocator_;

public:
  virtual void apply(Tensor t) = 0;
  void setAllocator(Ptr<Allocator> allocator) { allocator_ = allocator; }
};

/**
 * Use a lambda function of form [](Tensor t) { do something with t } to initalize tensor
 */
Ptr<NodeInitializer> lambda(std::function<void(Tensor)>&& func);

/**
 * Use a lambda function of form [](Tensor t) { do something with t } to initalize tensor
 * Create temporary tensor of Type intermediateType first, initialize and then copy/convert to actual Tensor
 * Useful for functions that can only operator on a specific type of tensor
 */
Ptr<NodeInitializer> lambda(std::function<void(Tensor)>&& func, Type intermediateType);

/**
 * Fill tensor with given value
 *
 * Creates a NodeInitializer that will fill the given tensor
 * with `value`. Works with any underlying numeric tensor type.
 *
 * @return A NodeInitializer which can be called on any tensor
 */
Ptr<NodeInitializer> fromValue(float value);

/**
 * Fill tensor with `0`
 *
 * Creates a NodeInitializer that will fill the given tensor
 * with `0`. Works with any underlying numeric tensor type.
 *
 * @return A NodeInitializer which can be called on any tensor
 */
Ptr<NodeInitializer> zeros();

/**
 * Fill tensor with `1`
 *
 * Creates a NodeInitializer that will fill the given tensor
 * with `1`. Works with any underlying numeric tensor type.
 *
 * @return A NodeInitializer which can be called on any tensor
 */
Ptr<NodeInitializer> ones();

/**
 * Set diagonal of two dimensional quadratic tensor to `value`.
 *
 * Sets all values of the tensor to 0 and fills the diagonal with
 * the given `value`. If no value is specified `1` is used by default.
 *
 * @return A NodeInitializer which can be called on any tensor
 */
Ptr<NodeInitializer> eye(float value = 1.f);

/**
 * Fill tensor with normally distributed random numbers
 *
 * Be default this generates floating point numbers from the
 * normal distribution N(0, 1) unless specified differently.
 *
 * If compiled with `CUDA`, `marian` will use the `cuRand` library
 * for both, GPU and CPU computation. The random sequences generated
 * are the same on both devices.
 *
 * If `marian` is compiled without `CUDA`, a random generator
 * from the C++ standard library is used. These random generators
 * do not have the same random sequences.
 *
 * @return A NodeInitializer which can be called on any tensor
 */
Ptr<NodeInitializer> normal(float mean = 0.f, float stddev = 1.f);

/**
 * Fill tensor with uniformly distributed random numbers
 *
 * Be default this generates floating point numbers from the
 * uniform distribution U(0, 1) unless specified differently.
 *
 * If compiled with `CUDA`, `marian` will use the `cuRand` library
 * for both, GPU and CPU computation. The random sequences generated
 * are the same on both devices.
 *
 * If `marian` is compiled without `CUDA`, a random generator
 * from the C++ standard library is used. These random generators
 * do not have the same random sequences.
 *
 * @return A NodeInitializer which can be called on any tensor
 */
Ptr<NodeInitializer> uniform(float a = 0.f, float b = 1.f);

Ptr<NodeInitializer> bernoulli(float p, float scale = 1.f, float shift = 0.f);
Ptr<NodeInitializer> glorotUniform(bool fanIn = false, bool fanOut = false, float scale = 1.f);
Ptr<NodeInitializer> glorotNormal(bool fanIn = false, bool fanOut = false, float scale = 1.f);

Ptr<NodeInitializer> dropout(float dropoutProbabilty);
Ptr<NodeInitializer> gumbel(float eps = 1e-5f);
Ptr<NodeInitializer> dummy();

template <typename T>
Ptr<NodeInitializer> fromVector(const std::vector<T>& v);
Ptr<NodeInitializer> fromSparseVector(std::pair<std::vector<size_t>, std::vector<float>>& v);
Ptr<NodeInitializer> fromItem(const io::Item& item);
Ptr<NodeInitializer> fromTensor(Tensor tensor);

Ptr<NodeInitializer> fromWord2vec(const std::string& file,
                                  int dimVoc,
                                  int dimEmb,
                                  bool normalize = false);

/**
 * Computes Google's sinusoidal position embeddings
 * starting from position 'start' taking into account
 * batch and time dimensions of the tensor.
 *
 * Expected tensor layout {-2: time, -1: model}
 *
 * Usually gets later reshaped to {time, 1, model} and
 * added with a broadcast to learned embeddings. Positional
 * embeddings are the same for each batch entry and change
 * over time.
 */
Ptr<NodeInitializer> sinusoidalPositionEmbeddings(int start);

}  // namespace inits

}  // namespace marian
