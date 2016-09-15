#pragma once

#include <random>
#include <algorithm>
#include <iterator>
#include <functional>

#include "tensor.h"

namespace marian {

void zeros(Tensor t) {
  t.set(0.f);
}

void ones(Tensor t) {
  t.set(1.0f);
}

template <class Distribution>
void distribution(Tensor t, float a, float b) {
  std::random_device device;
  std::default_random_engine engine(device());
  Distribution dist(a, b);
  auto gen = std::bind(dist, engine);

  std::vector<float> vals(t.size());
  std::generate(begin(vals), end(vals), gen);

  t << vals;
}

std::function<void(Tensor)> normal(float mean = 0.0, float std = 0.1) {
  return [mean, std](Tensor t) {
    distribution<std::normal_distribution<float>>(t, mean, std);
  }; 
}

std::function<void(Tensor)> uniform(float a = 0.0, float b = 0.1) {
  return [a, b](Tensor t) {
    distribution<std::uniform_real_distribution<float>>(t, a, b);
  };  
}

std::function<void(Tensor)> from_vector(const std::vector<float>& v) {
  return [v](Tensor t) {
    t << v;
  };
}
  
} // namespace marian
