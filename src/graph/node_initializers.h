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

NodeInitializer diag(float val);

template <class Distribution, class Iterator>
void distribution(Iterator begin, Iterator end, float a, float b) {
  std::default_random_engine engine((unsigned int)Config::seed++);
  Distribution dist(a, b);
  auto gen = std::bind(dist, engine);
  std::generate(begin, end, gen);
}

template <class Distribution>
void distribution(std::vector<float>& vals, float a, float b) {
  distribution<Distribution>(vals.begin(), vals.end(), a, b);
}

template <class Distribution>
void distribution(Tensor t, float a, float b) {
  std::vector<float> vals(t->size());
  distribution<Distribution>(vals.begin(), vals.end(), a, b);
  t->set(vals);
}

NodeInitializer normal(float scale = 0.1, bool ortho = true);

NodeInitializer uniform(float scale = 0.1);

static inline void dummy(Tensor t) {}

void ortho(Tensor t);

void glorot_uniform(Tensor t);

void xorshift(Tensor t);

void glorot_normal(Tensor t);

NodeInitializer from_vector(const std::vector<float>& v);
NodeInitializer from_vector(const std::vector<size_t>& v);

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
