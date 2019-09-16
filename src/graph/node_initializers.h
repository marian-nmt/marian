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
 * A NodeInitializer is a functor that is associated with parameters 
 * and constants, and is invoked on a tensor during node intialization. 
 * You need to override NodeIntializer::apply(Tensor) with your own 
 * functionality or use a fromLambda intializer.
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
Ptr<NodeInitializer> fromLambda(std::function<void(Tensor)>&& func);

/**
 * Use a lambda function of form [](Tensor t) { do something with t } to initalize tensor
 * Create temporary tensor of Type intermediateType first, initialize and then copy/convert to actual Tensor
 * Useful for functions that can only operate on a specific type of tensor
 */
Ptr<NodeInitializer> fromLambda(std::function<void(Tensor)>&& func, Type intermediateType);

/**
 * Initialize tensor with given value
 *
 * Creates a NodeInitializer that will intialize the given tensor
 * with `value`. Works with any underlying numeric tensor type.
 *
 * @return A NodeInitializer
 */
Ptr<NodeInitializer> fromValue(float value);

/**
 * Fill tensor with `0`
 *
 * Creates a NodeInitializer that will intialize the given tensor
 * with `0`. Works with any underlying numeric tensor type.
 *
 * @return A NodeInitializer
 */
static Ptr<NodeInitializer> zeros() { return fromValue(0.0f); }

/**
 * Fill tensor with `1`
 *
 * Creates a NodeInitializer that will intialize the given tensor
 * with `1`. Works with any underlying numeric tensor type.
 *
 * @return A NodeInitializer
 */
static Ptr<NodeInitializer> ones() { return fromValue(1.0f); }

/**
 * Set diagonal of two dimensional quadratic matrix to `value`.
 *
 * Sets all values of the tensor to 0 and intializes the diagonal with
 * the given `value`. If no value is specified `1` is used by default.
 *
 * @return A NodeInitializer
 */
Ptr<NodeInitializer> eye(float value = 1.f);

/**
 * Intialize tensor with normally distributed random numbers
 *
 * Be default this generates floating point numbers from the
 * normal distribution Normal(0, 1) unless specified differently.
 *
 * If compiled with `CUDA`, `marian` will use the `cuRand` library
 * for both, GPU and CPU computation. The random sequences generated
 * are the same on both devices.
 *
 * If `marian` is compiled without `CUDA`, a random generator
 * from the C++ standard library is used. These random generators
 * do not have the same random sequences.
 *
 * @return A NodeInitializer
 */
Ptr<NodeInitializer> normal(float mean = 0.f, float stddev = 1.f);

/**
 * Intialize tensor with uniformly distributed random numbers
 *
 * Be default this generates floating point numbers from the
 * uniform distribution Uniform(0, 1) unless specified differently.
 *
 * If compiled with `CUDA`, `marian` will use the `cuRand` library
 * for both, GPU and CPU computation. The random sequences generated
 * are the same on both devices.
 *
 * If `marian` is compiled without `CUDA`, a random generator
 * from the C++ standard library is used. These random generators
 * do not have the same random sequences.
 *
 * @return A NodeInitializer
 */
Ptr<NodeInitializer> uniform(float a = 0.f, float b = 1.f);

// @TODO: add documentation
Ptr<NodeInitializer> bernoulli(float p, float scale = 1.f, float shift = 0.f);

// @TODO: add documentation
Ptr<NodeInitializer> glorotUniform(bool fanIn = false, bool fanOut = false, float scale = 1.f);

// @TODO: add documentation
Ptr<NodeInitializer> glorotNormal(bool fanIn = false, bool fanOut = false, float scale = 1.f);

// @TODO: add documentation
Ptr<NodeInitializer> dropout(float dropoutProbabilty);

/**
 * Intialize with gumbel noise, i.e. -log(-log(u)) where u ~ Uniform(0 + eps, 1 - eps)
 * 
 * @return A NodeInitializer
 */
Ptr<NodeInitializer> gumbel(float eps = 1e-5f);

// @TODO: add documentation
template <typename T>
Ptr<NodeInitializer> fromVector(const std::vector<T>& v);

// @TODO: add documentation
Ptr<NodeInitializer> fromSparseVector(std::pair<std::vector<size_t>, std::vector<float>>& v);

// @TODO: add documentation
Ptr<NodeInitializer> fromItem(const io::Item& item);

// @TODO: add documentation
Ptr<NodeInitializer> fromTensor(Tensor tensor);

// @TODO: add documentation
Ptr<NodeInitializer> fromWord2vec(const std::string& file,
                                  int dimVoc,
                                  int dimEmb,
                                  bool normalize = false);

/**
 * Computes Google's Transformer-style sinusoidal position embeddings
 * starting from position 'start' taking into account batch and time 
 * dimensions of the tensor.
 *
 * Expected tensor layout {-2: time, -1: model}
 *
 * Usually gets later reshaped to {time, 1, model} and
 * added with a broadcast to learned embeddings. Positional
 * embeddings are the same for each batch entry and change
 * over time steps.
 */
Ptr<NodeInitializer> sinusoidalPositionEmbeddings(int start);

}  // namespace inits

}  // namespace marian
