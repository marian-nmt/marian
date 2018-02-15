#pragma once

#include <functional>
#include <random>

#include "cnpy/cnpy.h"
#include "common/config.h"
#include "tensors/tensor.h"

namespace marian {

namespace inits {

float xor128();

// Use a constant seed for deterministic behaviour.
// std::default_random_engine engine(42);

void zeros(Tensor t);

void ones(Tensor t);

std::function<void(Tensor)> from_value(float v);

std::function<void(Tensor)> diag(float val);

template <class Distribution>
void distribution(std::vector<float>& vals, float a, float b) {
  std::default_random_engine engine(Config::seed++);
  Distribution dist(a, b);
  auto gen = std::bind(dist, engine);
  std::generate(begin(vals), end(vals), gen);
}

template <class Distribution>
void distribution(Tensor t, float a, float b) {
  std::vector<float> vals(t->size());
  distribution<Distribution>(vals, a, b);
  t << vals;
}

std::function<void(Tensor)> normal(float scale = 0.1, bool ortho = true);

std::function<void(Tensor)> uniform(float scale = 0.1);

void ortho(Tensor t);

void glorot_uniform(Tensor t);

void xorshift(Tensor t);

void glorot_normal(Tensor t);

std::function<void(Tensor)> from_vector(const std::vector<float>& v);
std::function<void(Tensor)> from_vector(const std::vector<size_t>& v);

std::function<void(Tensor)> from_sparse_vector(
    std::pair<std::vector<size_t>, std::vector<float>>& v);

std::function<void(Tensor)> from_numpy(const cnpy::NpyArray& np);

std::function<void(Tensor)> from_word2vec(const std::string& file,
                                          int dimVoc,
                                          int dimEmb,
                                          bool normalize = false);
}

}  // namespace marian
