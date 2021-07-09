// TODO: move to backend, into graph/
#pragma once

#include "common/config.h"
#include "tensors/tensor.h"
#include "tensors/tensor_operators.h"

#include <functional>
#include <random>

namespace marian {

class ExpressionGraph; // Forward declaration
/**
 * The namespace inits.
 * Declare class NodeInitializer and all the available functions to initialise a node.
*/
namespace inits {

/**
 * Base class for specialized NodeInitializers.
 * A NodeInitializer is a functor that is associated with parameters
 * and constants, and is invoked on a tensor during node initialization.
 * You need to override NodeInitializer::apply(Tensor) with your own
 * functionality or use a fromLambda initializer.
 * See node_initializers.cpp for examples.
 */
class NodeInitializer {
protected:
  Weak<Allocator> allocator_;

public:
  virtual void apply(Tensor t) = 0;
  void setAllocator(Ptr<Allocator> allocator) { allocator_ = allocator; }
  virtual ~NodeInitializer() {}
};

/**
 * Dummy do-nothing initializer. Mostly for testing.
 */
Ptr<NodeInitializer> dummy();

/**
 * Use a lambda function of form [](Tensor t) { do something with t } to initialize tensor.
 * @param func functor
 */
Ptr<NodeInitializer> fromLambda(std::function<void(Tensor)>&& func);

/**
 * Use a lambda function of form [](Tensor t) { do something with t } to initialize tensor.
 * Create temporary tensor of Type intermediateType first, initialize and then copy/convert to actual Tensor.
 * Useful for functions that can only operate on a specific type of tensor.
 */
Ptr<NodeInitializer> fromLambda(std::function<void(Tensor)>&& func, Type intermediateType);

/**
 * Initialize tensor with given value.
 * Creates a NodeInitializer that will initialize the given tensor
 * with `value`. Works with any underlying numeric tensor type.
 * @return A NodeInitializer
 */
Ptr<NodeInitializer> fromValue(float value);

/**
 * Fill tensor with `0`.
 * Creates a NodeInitializer that will initialize the given tensor
 * with `0`. Works with any underlying numeric tensor type.
 * @return A NodeInitializer
 */
static Ptr<NodeInitializer> zeros() { return fromValue(0.0f); }

/**
 * Fill tensor with `1`.
 * Creates a NodeInitializer that will initialize the given tensor
 * with `1`. Works with any underlying numeric tensor type.
 * @return A NodeInitializer
 */
static Ptr<NodeInitializer> ones() { return fromValue(1.0f); }

/**
 * Set diagonal of two dimensional quadratic matrix to `value`.
 * Sets all values of the tensor to 0 and initializes the diagonal with
 * the given `value`. If no value is specified `1` is used by default.
 * @return A NodeInitializer
 */
Ptr<NodeInitializer> eye(float value = 1.f);

/**
 * Initialize tensor with normally distributed random numbers.
 * By default this generates floating point numbers from the
 * normal distribution Normal(0, 1) unless specified differently.
 * If compiled with `CUDA`, `marian` will use the `cuRand` library
 * for both, GPU and CPU computation. The random sequences generated
 * are the same on both devices.
 * If `marian` is compiled without `CUDA`, a random generator
 * from the C++ standard library is used. These random generators
 * do not have the same random sequences.
 * @return A NodeInitializer
 */
Ptr<NodeInitializer> normal(float mean = 0.f, float stddev = 1.f);

/**
 * Initialize tensor with uniformly distributed random numbers.
 * By default this generates floating point numbers from the
 * uniform distribution Uniform(0, 1) unless specified differently.
 * If compiled with `CUDA`, `marian` will use the `cuRand` library
 * for both, GPU and CPU computation. The random sequences generated
 * are the same on both devices.
 * If `marian` is compiled without `CUDA`, a random generator
 * from the C++ standard library is used. These random generators
 * do not have the same random sequences.
 * @param a the lower bound of interval
 * @param b the upper bound of interval
 * @return A NodeInitializer
 */
Ptr<NodeInitializer> uniform(float a = 0.f, float b = 1.f);

/**
 * Initialize tensor with random numbers from Bernoulli Distribution.
 * The Bernoulli distribution is the discrete probability distribution of
 * a random variable which takes value `1` with probability p, and
 * value `0` with probability (1-p).
 * By default this function generates a tensor of 0 and 1 with probability p
 * if bernoulli(p) is called. We offer `scale` and `shift` parameters which
 * can map {0,1} to {0,1}*`scale`+`shift`.
 * E.g., bernoulli(tensor, 0.5f, 2.f, -1.f) where p=0.5f, scale=2.f, shift=-1.f.
 * {0,1} is mapped to {0,1}*2+(-1)= {-1,1}. It generates a tensor composed of
 * 50% of 1 and 50% of -1.
 * @return A NodeInitializer
 */
Ptr<NodeInitializer> bernoulli(float p, float scale = 1.f, float shift = 0.f);

/**
 * Initialize tensor with random numbers from Glorot uniform distribution.
 * The <a href=http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf>Glorot uniform</a>,
 * also called Xavier uniform, is designed to keep the scale of
 * the gradients roughly the same in all layers.
 * This function offers three variants (modes).
 * The values of the tensor is sampled from Uniform(-x*scale, x*scale):
 *   - when fanIn=false and fanOut=false (by default):
 *      x = sqrt(6 / (in + out))
 *   - when fanIn=true and fanOut=false (fanIn mode):
 *      x = sqrt(3 / in)
 *   - when fanIn=false and fanOut=false (fanOut mode):
 *      x = sqrt(3 / out)
 * where `in` is the number of input units in the tensor, `out` is the number of output units.
 * `scale` is used to change the range of Uniform distribution.
 * @return A NodeInitializer
 */
Ptr<NodeInitializer> glorotUniform(bool fanIn = false, bool fanOut = false, float scale = 1.f);

/**
 * Initialize tensor with random numbers from Glorot Normal distribution.
 * Similar to function glorotUniform(), this function adopts Normal distribution instead of
 * uniform distribution.
 * This function offers three variants (modes).
 * The values of the tensor is sampled from Normal(-x*scale, x*scale):
 *   - when fanIn=false and fanOut=false (by default):
 *      x = sqrt(2 / (in + out))
 *   - when fanIn=true and fanOut=false (fanIn mode):
 *      x = sqrt(1 / in)
 *   - when fanIn=false and fanOut=false (fanOut mode):
 *      x = sqrt(1 / out)
 * where `in` is the number of input units in the tensor, `out` is the number of output units.
 * `scale` is used to change the range of Normal distribution.
 * @return A NodeInitializer
 */
Ptr<NodeInitializer> glorotNormal(bool fanIn = false, bool fanOut = false, float scale = 1.f);

/**
 * Initialize a dropout mask (a tensor of 0 and 1) with given dropout probability.
 * <a href=https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf>Dropout</a>
 * is proposed as a technique to prevent Neural Networks from overfitting.
 * @param dropoutProbability a float type defines the dropout probability.
 *        E.g., dropoutProbability=0.1 means 90% of values are kept.
 * @return A NodeInitializer
 */
Ptr<NodeInitializer> dropout(float dropoutProbability);

/**
 * Initialize with gumbel noise, i.e. -log(-log(u)) where u ~ Uniform(0 + eps, 1 - eps).
 * @param eps a variable protects from log(0)
 * @return A NodeInitializer
 */
Ptr<NodeInitializer> gumbel(float eps = 1e-5f);

/**
 * Initialize tensor by *copying* from the given vector.
 * Creates a NodeInitializer that will initialize the tensor
 * by *copying* the values from the given vector
 * @param v vector
 * @return A NodeInitializer
 */
template <typename T>
Ptr<NodeInitializer> fromVector(const std::vector<T>& v);

/**
 * Initialize tensor by *moving* from the given vector.
 * Creates a NodeInitializer that will initialize the tensor by *moving* the values
 * from the given vector into this tensor, and the given vector may be emptied.
 * This version is the <a href=https://en.cppreference.com/w/cpp/language/reference>
 * rvalue reference</a> overloading.
 * @param v vector
 * @return A NodeInitializer
 */
template <typename T>
Ptr<NodeInitializer> fromVector(std::vector<T>&& v);

/**
 * Initialize tensor from a given sparse vector.
 * Creates a NodeInitializer that will initialize the tensor from a given
 * sparse vector (stored in std::pair). The resulting tensor is first filled
 * with `1e-6` (a placeholder for non-zero element), then set the value to
 * the given sparse vector.
 * @param v the sparse vector is stored in `std::pair`:
 *   - the first object (v.first) holds the indexes (in a vector)
 *   - the second object (v.second) holds the corresponding values (in a vector).
 *   This means the value of the resulting tensor at index v.first[i] is v.second[i].
 * @return A NodeInitializer
 */
Ptr<NodeInitializer> fromSparseVector(std::pair<std::vector<size_t>, std::vector<float>>& v);

/**
 * Initialize tensor by copying from the given io::Item.
 * Creates a NodeInitializer that will initialize the tensor by copying the values
 * from the given io::Item. If this io::Item is a memory-mapped item, then the
 * function will set the memory region pointing to this item. If this io::Item is
 * a regular item, then the function will copy the values from this item.
 * @return A NodeInitializer
 */
Ptr<NodeInitializer> fromItem(const io::Item& item);

/**
 * Initialize tensor by copying from the given tensor.
 * Creates a NodeInitializer that will initialize the tensor
 * by copying the values from the given tensor.
 * @return A NodeInitializer
 */
Ptr<NodeInitializer> fromTensor(Tensor tensor);

/**
 * Initialize tensor from a file.
 * Creates a NodeInitializer that will initialize the tensor
 * by copying the values from the given file. This function is
 * mainly used for loading embedding vectors from a file.
 * @param file filename
 * @param dimVoc the number of words in the vocabulary
 * @param dimEmb the length of embedding vectors
 * @param normalize a flag holds whether the values are normalize.
 * Here we adopt the method of <a
 * href=https://en.wikipedia.org/wiki/Feature_scaling#Scaling_to_unit_length>
 * scaling to unit length</a>, i.e., dividing each element by the Euclidean length of the vector.
 * @return A NodeInitializer
 */
Ptr<NodeInitializer> fromWord2vec(const std::string& file,
                                  int dimVoc,
                                  int dimEmb,
                                  bool normalize = false);

/**
 * Computes Google's sinusoidal position embeddings.
 * Computes Google's Transformer-style sinusoidal position embeddings
 * starting from position 'start' taking into account batch and time
 * dimensions of the tensor. Expected tensor layout {-2: time, -1: model}.
 * Usually gets later reshaped to {time, 1, model} and added with a broadcast
 * to learned embeddings. Positional embeddings are the same for each batch
 * entry and change over time steps.
 */
Ptr<NodeInitializer> sinusoidalPositionEmbeddings(int start);

/**
 * Computes the equivalent of Python's range().
 * Computes a range from begin to end-1, like Python's range().
 * The constant being initialized must have one dimension that matches
 * the number of elements being generated, while any other dimension must be 1.
 */
template <typename T>
Ptr<NodeInitializer> range(T begin, T end, T step = (T)1);

}  // namespace inits

}  // namespace marian
