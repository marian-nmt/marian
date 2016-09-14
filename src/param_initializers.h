#pragma once

#include <random>
#include <algorithm>
#include <iterator>
#include <functional>

#include "tensor.h"

namespace marian {

void zeros(Tensor t) {
  std::vector<float> vals(t.size(), 0.0f);
  thrust::copy(vals.begin(), vals.end(), t.begin());
}

void ones(Tensor t) {
  std::vector<float> vals(t.size(), 1.0f);
  thrust::copy(vals.begin(), vals.end(), t.begin());
}

void randreal(Tensor t) {
  std::random_device device;
  std::default_random_engine engine(device());
  std::uniform_real_distribution<> dist(0, 1);
  auto gen = std::bind(dist, engine);

  std::vector<float> vals(t.size());
  std::generate(begin(vals), end(vals), gen);

  thrust::copy(vals.begin(), vals.end(), t.begin());
}

} // namespace marian
