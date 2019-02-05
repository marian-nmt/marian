#include "graph/node_initializers.h"
#include "layers/word2vec_reader.h"
#include "tensors/tensor_operators.h"

#include <stdint.h>
#include <algorithm>
#include <iterator>
#include <random>

namespace marian {

namespace inits {

void zeros(Tensor t) {
  t->set(0.f);
}

void ones(Tensor t) {
  t->set(1.0f);
}

NodeInitializer from_value(float v) {
  return [v](Tensor t) { t->set(v); };
}

// diagonal matrix with value val along diagonal
NodeInitializer eye(float val) {
  return [val](Tensor t) {
    ABORT_IF(t->shape().size() != 2 || t->shape()[-1] != t->shape()[-2],
             "eye(val) is defined only for quadratic tensors, shape is {}",
             t->shape());

    // @TODO: implement efficient version on the GPU
    std::vector<float> vec(t->size(), 0);
    for(int i = 0; i < t->shape()[-1]; ++i)
      vec[i * t->shape()[0] + i] = val;
    t->set(vec);
  };
}

NodeInitializer uniform(float a, float b) {
  return [a, b](Tensor tensor) {
    tensor->getBackend()->getRandomGenerator()->uniform(tensor, a, b);
  };
}

NodeInitializer normal(float mean, float stddev) {
  return [mean, stddev](Tensor tensor) {
    tensor->getBackend()->getRandomGenerator()->normal(tensor, mean, stddev);
  };
}

// @TODO: replace with parametered version below
void glorot_uniform(Tensor tensor) {
  float scale = sqrtf(6.0f / (tensor->shape()[-2] + tensor->shape()[-1]));
  uniform(-scale, scale)(tensor);
}

NodeInitializer glorot_uniform2(bool fanIn, bool fanOut) {
  return [fanIn, fanOut](Tensor tensor) {
    float scale = 1.f;
    if(fanIn && fanOut)
      scale = sqrtf(6.0f / (tensor->shape()[-2] + tensor->shape()[-1]));
    else if(!fanIn && fanOut)
      scale = sqrtf(3.0f / tensor->shape()[-1]);
    else if(fanIn && !fanOut)
      scale = sqrtf(3.0f / tensor->shape()[-2]);
    else
      ABORT("You need to set fanIn or fanOut or both to true");

    uniform(-scale, scale)(tensor);
  };
}

// @TODO: replace with parametered version below
void glorot_normal(Tensor tensor) {
  float scale = sqrtf(2.0f / (tensor->shape()[-2] + tensor->shape()[-1]));
  normal(0.f, scale)(tensor);
}

NodeInitializer glorot_normal2(bool fanIn, bool fanOut) {
  return [fanIn, fanOut](Tensor tensor) {
    float scale = 1.f;
    if(fanIn && fanOut)
      scale = sqrtf(2.0f / (tensor->shape()[-2] + tensor->shape()[-1]));
    else if(!fanIn && fanOut)
      scale = sqrtf(1.0f / tensor->shape()[-1]);
    else if(fanIn && !fanOut)
      scale = sqrtf(1.0f / tensor->shape()[-2]);
    else
      ABORT("You need to set fanIn or fanOut or both to true");

    normal(0.f, scale)(tensor);
  };
}

NodeInitializer bernoulli(float prob, float scale) {
  return [prob, scale](Tensor tensor) {
    Bernoulli(tensor, prob, scale);
  };
}

NodeInitializer dropout(float dropProb) {
  return [dropProb](Tensor t) {
    Dropout(t, dropProb);
  };
}

// gumbel noise:
// -log(-log(uniform(0.f + eps, 1.f - eps)));
void gumbel(Tensor tensor) {
  using namespace functional;
  // @TODO: make eps a parameter? Seems to influence amplitude quite heavily
  float eps = 1e-05f;
  uniform(0.f + eps, 1.f - eps)(tensor);
  Element(_1 = -log(-log(_1)), tensor);
}

NodeInitializer from_vector(const std::vector<float>& v) {
  auto vPtr = New<std::vector<float>>(v.begin(), v.end());
  return
      [vPtr](Tensor t) { t->set(vPtr->data(), vPtr->data() + vPtr->size()); };
}

// @TODO: handle this better with proper type support, the NodeInitializer
// should be able to inform the calling function about the tensor type it
// is expecting. Probably needs to turn into struct with type information.
NodeInitializer from_vector(const std::vector<IndexType>& v) {
  auto vPtr = New<std::vector<IndexType>>(v.begin(), v.end());
  return
      [vPtr](Tensor t) { t->set(vPtr->data(), vPtr->data() + vPtr->size()); };
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

// Computes Google's sinusoidal position embeddings
NodeInitializer sinusoidalPositionEmbeddings(int start) {
  return [start](Tensor t) {
    int dimEmb   = t->shape()[-1];
    int dimWords = (int)t->size() / dimEmb;

    float numTimescales = (float)dimEmb / 2;
    float logTimescaleIncrement = std::log(10000.f) / (numTimescales - 1.f);

    std::vector<float> vPos(dimEmb * dimWords, 0);
    for(int p = start; p < dimWords + start; ++p) {
      for(int i = 0; i < numTimescales; ++i) {
        float v = p * std::exp(i * -logTimescaleIncrement);
        vPos[(p - start) * dimEmb + i                     ] = std::sin(v);
        vPos[(p - start) * dimEmb + (int)numTimescales + i] = std::cos(v); // @TODO: is int vs. float correct for num_timescales?
      }
    }

    from_vector(vPos)(t);
  };
}

}  // namespace inits

}  // namespace marian
