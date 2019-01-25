// TODO: move to backend, into graph/
#pragma once

#include "common/config.h"
#include "tensors/tensor.h"

#include <functional>
#include <random>

namespace marian {

typedef std::function<void(Tensor)> NodeInitializer;

namespace inits {

void zeros(Tensor t);

void ones(Tensor t);

NodeInitializer from_value(float v);

NodeInitializer eye(float val = 1.f);

NodeInitializer normal(float mean = 0.f, float stddev = 1.f);

NodeInitializer uniform(float a = 0.f, float b = 1.f);

void glorot_uniform(Tensor t);
NodeInitializer glorot_uniform2(bool fanIn = true, bool fanOut = true);

void glorot_normal(Tensor t);
NodeInitializer glorot_normal2(bool fanIn = true, bool fanOut = true);

NodeInitializer bernoulli(float p, float scale = 1.f);

NodeInitializer dropout(float dropProb);

void gumbel(Tensor t);

static inline void dummy(Tensor) {}

NodeInitializer from_vector(const std::vector<float>& v);
NodeInitializer from_vector(const std::vector<IndexType>& v);

NodeInitializer from_item(const io::Item& item);

NodeInitializer from_sparse_vector(
    std::pair<std::vector<size_t>, std::vector<float>>& v);

// NodeInitializer from_numpy(const cnpy::NpyArrayPtr& np);

NodeInitializer from_word2vec(const std::string& file,
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
NodeInitializer sinusoidalPositionEmbeddings(int start);

}  // namespace inits

}  // namespace marian
