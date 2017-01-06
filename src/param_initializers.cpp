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
#include <stdint.h>

#include "param_initializers.h"

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
//std::default_random_engine engine(42);

void zeros(Tensor t) {
  t->set(0.f);
}

void ones(Tensor t) {
  t->set(1.0f);
}

std::function<void(Tensor)> from_value(float v) {
  return [v](Tensor t) {
    t->set(v);
  };
}

std::function<void(Tensor)> diag(float val) {
  return [val](Tensor t) {
    if(t->shape()[0] == t->shape()[1] && t->shape()[2] == 1 && t->shape()[3] == 1) {
      std::vector<float> vec(t->size(), 0);
      for(int i = 0; i < t->shape()[0]; ++i)
        vec[i * t->shape()[1] + i] = val;
      t->set(vec);
    }
  };
}

std::function<void(Tensor)> normal(float mean, float std) {
  return [mean, std](Tensor t) {
    distribution<std::normal_distribution<float>>(t, mean, std);
  };
}

std::function<void(Tensor)> uniform(float a, float b) {
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
    t->set(v);
  };
}

std::function<void(Tensor)> from_numpy(const cnpy::NpyArray& np) {
  size_t size = 1;
  for(int i = 0; i < np.shape.size(); ++i) {
    size *= np.shape[i];
  };

  std::vector<float> npv(size);
  std::copy((float*)np.data, (float*)np.data + size, npv.begin());

  return [npv](Tensor t) {
    t->set(npv);
  };
}

} // namespace marian
