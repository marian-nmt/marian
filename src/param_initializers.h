#pragma once

// This file is part of the Marian toolkit.
// Marian is copyright (c) 2016 Marcin Junczys-Dowmunt.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <random>
#include <algorithm>
#include <iterator>
#include <functional>
#include <stdint.h>

#include "tensors/tensor.h"

namespace marian {

float xor128() {
    static uint64_t x = 123456789;
    static uint64_t y = 362436069;
    static uint64_t z = 521288629;
    static uint64_t w = 88675123;
    uint64_t t;

    t = (x ^ (x << 11)) % 1000;
    x = y; y = z; z = w;
    w = (w ^ (w >> 19) ^ t ^ (t >> 8)) % 1000;
    return 0.1 * ((w % 1000) / 1000.f) - 0.05;
}

// Use a constant seed for deterministic behaviour.
std::default_random_engine engine(42);

void zeros(Tensor t) {
  t->set(0.f);
}

void ones(Tensor t) {
  t->set(1.0f);
}

template <class Distribution>
void distribution(Tensor t, float a, float b) {
  //std::random_device device;
  //std::default_random_engine engine(device());
  Distribution dist(a, b);
  auto gen = std::bind(dist, engine);

  std::vector<float> vals(t->size());
  std::generate(begin(vals), end(vals), gen);

  t << vals;
}

std::function<void(Tensor)> normal(float mean = 0.0, float std = 0.05) {
  return [mean, std](Tensor t) {
    distribution<std::normal_distribution<float>>(t, mean, std);
  };
}

std::function<void(Tensor)> uniform(float a = -0.05, float b = 0.05) {
  return [a, b](Tensor t) {
    distribution<std::uniform_real_distribution<float>>(t, a, b);
  };
}

void glorot_uniform(Tensor t) {
  float b = sqrtf( 6.0f / (t->shape()[0] + t->shape()[1]) );
  distribution<std::uniform_real_distribution<float>>(t, -b, b);
}

void xorshift(Tensor t) {
  std::vector<float> vals(t->size());
  for(auto&& v : vals)
    v = xor128();
  t << vals;
}

void glorot_normal(Tensor t) {
  float b = sqrtf( 2.0f / (t->shape()[0] + t->shape()[1]) );
  distribution<std::uniform_real_distribution<float>>(t, -b, b);
}

std::function<void(Tensor)> from_vector(const std::vector<float>& v) {
  return [v](Tensor t) {
    t << v;
  };
}

} // namespace marian
