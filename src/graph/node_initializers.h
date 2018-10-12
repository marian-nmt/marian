// TODO: move to backend, into graph/
#pragma once

#include "common/config.h"
#include "tensors/tensor.h"

#include <functional>
#include <random>

namespace marian {

typedef std::function<void(Tensor)> NodeInitializer;

namespace inits {

float xor128();

// Use a constant seed for deterministic behaviour.
// std::default_random_engine engine(42);

void zeros(Tensor t);

void ones(Tensor t);

NodeInitializer from_value(float v);

NodeInitializer eye(float val = 1.f);

NodeInitializer normal(float mju, float sigma);

NodeInitializer uniform(float a, float b);

NodeInitializer dropout(float prob);

void gumbel(Tensor t);

static inline void dummy(Tensor t) {}

void glorot_uniform(Tensor t);

void xorshift(Tensor t);

void glorot_normal(Tensor t);

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
