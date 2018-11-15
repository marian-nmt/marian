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

void glorot_normal(Tensor t);

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
}  // namespace inits

}  // namespace marian
