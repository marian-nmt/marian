#include "graph/node_initializers.h"
#include "3rd_party/svd/svd.h"
#include "layers/word2vec_reader.h"

#include <stdint.h>
#include <algorithm>
#include <iterator>
#include <random>

namespace marian {

namespace inits {

float xor128() {
  static uint64_t x = 123456789;
  static uint64_t y = 362436069;
  static uint64_t z = 521288629;
  static uint64_t w = 88675123;
  uint64_t t;

  t = (x ^ (x << 11)) % 1000;
  x = y;
  y = z;
  z = w;
  w = (w ^ (w >> 19) ^ t ^ (t >> 8)) % 1000;
  return 0.1f * ((w % 1000) / 1000.f) - 0.05f;
}

void zeros(Tensor t) {
  t->set(0.f);
}

void ones(Tensor t) {
  t->set(1.0f);
}

NodeInitializer from_value(float v) {
  return [v](Tensor t) { t->set(v); };
}

NodeInitializer diag(float val) {
  return [val](Tensor t) {
    if(t->shape()[0] == t->shape()[1] && t->shape()[2] == 1
       && t->shape()[3] == 1) {
      std::vector<float> vec(t->size(), 0);
      for(int i = 0; i < t->shape()[0]; ++i)
        vec[i * t->shape()[1] + i] = val;
      t->set(vec);
    }
  };
}

NodeInitializer normal(float scale, bool /*ortho*/ /*= true*/) {
  return [scale](Tensor t) {
    distribution<std::normal_distribution<float>>(t, 0, scale);
  };
}

NodeInitializer uniform(float scale) {
  return [scale](Tensor t) {
    distribution<std::uniform_real_distribution<float>>(t, -scale, scale);
  };
}

void glorot_uniform(Tensor t) {
  float scale = sqrtf(6.0f / (t->shape()[-2] + t->shape()[-1]));
  distribution<std::uniform_real_distribution<float>>(t, -scale, scale);
}

void xorshift(Tensor t) {
  std::vector<float> vals(t->size());
  for(auto&& v : vals)
    v = xor128();
  t->set(vals);
}

void glorot_normal(Tensor t) {
  float scale = sqrtf(2.0f / (t->shape()[-2] + t->shape()[-1]));
  distribution<std::normal_distribution<float>>(t, 0, scale);
}

void svd(std::vector<float>& vec, Shape shape) {
  int rows = shape[0] * shape[2] * shape[3];
  int cols = shape[1];

  int n = std::min(rows, cols);
  int m = std::max(rows, cols);

  ABORT_IF(m % n != 0,
           "Matrix dimensions must be equal or multiples of each other");

  for(int i = 0; i < shape.elements(); i += n * n) {
    std::vector<float> t1(n);
    std::vector<float> t2(n * n);
    float* a = vec.data() + i;
    float* w = t1.data();
    float* v = t2.data();
    dsvd(a, n, n, w, v);
  }
}

void ortho(Tensor t) {
  std::vector<float> vec(t->size());
  distribution<std::normal_distribution<float>>(vec, 0, 1);
  svd(vec, t->shape());
  t->set(vec);
}

NodeInitializer from_vector(const std::vector<float>& v) {
  auto vPtr = New<std::vector<float>>(v.begin(), v.end());
  return
      [vPtr](Tensor t) { t->set(vPtr->data(), vPtr->data() + vPtr->size()); };
}

NodeInitializer from_vector(const std::vector<size_t>& v) {
  auto n = v.size();
  std::vector<float> vf(n);
  for (size_t i = 0; i < n; i++)
    vf[i] = (float)v[i];
  return from_vector(vf);
}

NodeInitializer from_sparse_vector(
    std::pair<std::vector<size_t>, std::vector<float>>& v) {
  return [v](Tensor t) {
    t->set(1e-6);
    t->setSparse(v.first, v.second);
  };
}

// NodeInitializer from_numpy(const cnpy::NpyArrayPtr& np) {
//  return [np](Tensor t) {
//    size_t size = 1;
//    for(size_t dim : np->shape)
//      size *= dim;
//    t->set((float*)np->data(), (float*)np->data() + size);
//  };
//}

// move this somewhere else
NodeInitializer from_word2vec(const std::string& file,
                              int dimVoc,
                              int dimEmb,
                              bool normalize /*= false*/) {
  return [file, dimVoc, dimEmb, normalize](Tensor t) {
    auto embs = Word2VecReader().read(file, dimVoc, dimEmb);

    if(normalize) {
      float norm = 0;
      for(auto e : embs)
        norm += e * e;
      norm = std::sqrt(norm);
      if(norm != 0)
        for(auto& e : embs)
          e = e / norm;
    }
    t->set(embs);
  };
}

NodeInitializer from_item(const io::Item& item) {
  if(item.mapped) {
    return [item](Tensor t) {
      // @TODO: implement other types, for now croak loudly.
      ABORT_IF(t->getBackend()->getDeviceId().type != DeviceType::cpu,
               "Memory mapping only works for CPU tensors");
      ABORT_IF(!matchType<float>(t->type()),
               "Tensor type and type for mapping do not match");
      auto mp = New<MemoryPiece>((uint8_t*)item.ptr, t->size() * sizeof(float));
      t->reset(mp);
    };
  } else {
    return [item](Tensor t) {
      // @TODO: implement other types, for now croak loudly.
      ABORT_IF(!matchType<float>(t->type()),
               "Tensor type and type for mapping do not match");
      t->set((const float*)item.bytes.data(),
             (const float*)item.bytes.data() + t->size());
    };
  }
}

}  // namespace inits

}  // namespace marian
