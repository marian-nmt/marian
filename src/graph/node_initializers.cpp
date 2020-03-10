#include "graph/node_initializers.h"
#include "layers/word2vec_reader.h"
#include "tensors/tensor_operators.h"

#include <stdint.h>
#include <algorithm>
#include <iterator>
#include <random>

namespace marian {

namespace inits {

class LambdaInit : public NodeInitializer {
  private:
    std::function<void(Tensor)> lambda_;

  public:
    LambdaInit(std::function<void(Tensor)>&& lambda) : lambda_(std::move(lambda)) {}

    void apply(Tensor tensor) override {
      lambda_(tensor);
    }
};

class LambdaInitConvert : public NodeInitializer {
  private:
    std::function<void(Tensor)> lambda_;
    Type intermediateType_; // is used for the creation of a temporary intermedia tensor on which the lambda actually operates.
                            // This tensor is then automatically cast and copied to the type of the actual tensor. 

  public:
    LambdaInitConvert(std::function<void(Tensor)>&& lambda,
                      Type intermediateType)
      : lambda_(std::move(lambda)), intermediateType_(intermediateType) {}

    void apply(Tensor tensor) override {
      if(tensor->type() != intermediateType_) {
        auto sharedAllocator = allocator_.lock();
        ABORT_IF(!sharedAllocator, "Allocator in LambdaInitConvert has not been set or expired");

        auto memory = sharedAllocator->alloc(requiredBytes(tensor->shape(), intermediateType_));
        auto temp = TensorBase::New(memory,
                                    tensor->shape(),
                                    intermediateType_,
                                    tensor->getBackend());
        lambda_(temp);
        CopyCast(tensor, temp); // Cast and copy from temp to tensor
        sharedAllocator->free(memory);
      }
      else {
        lambda_(tensor);
      }
    }
};

Ptr<NodeInitializer> fromLambda(std::function<void(Tensor)>&& func) {
  return New<LambdaInit>(std::move(func));
}

Ptr<NodeInitializer> fromLambda(std::function<void(Tensor)>&& func, Type intermediateType) {
  return New<LambdaInitConvert>(std::move(func), intermediateType);
}

Ptr<NodeInitializer> fromValue(float v) {
  return fromLambda([v](Tensor t){ t->set(v); });
}

// diagonal matrix with value val along diagonal
Ptr<NodeInitializer> eye(float val) {
  auto eyeLambda = [val](Tensor t) {
    ABORT_IF(t->shape().size() != 2 || t->shape()[-1] != t->shape()[-2],
              "eye(val) is defined only for quadratic tensors, shape is {}",
              t->shape());

    // @TODO: implement efficient version on the GPU
    std::vector<float> vec(t->size(), 0);
    for(int i = 0; i < t->shape()[-1]; ++i)
      vec[i * t->shape()[0] + i] = val;

    t->set(vec);
  };

  return fromLambda(eyeLambda, Type::float32);
}

Ptr<NodeInitializer> uniform(float a, float b) {
  // only works for float, hence the conversion through intermedia type Type::float32
  return fromLambda([a, b](Tensor t) { t->getBackend()->getRandomGenerator()->uniform(t, a, b); }, Type::float32);
}

Ptr<NodeInitializer> normal(float mean, float stddev) {
  // only works for float, hence the conversion through intermedia type Type::float32
  return fromLambda([mean, stddev](Tensor t) { t->getBackend()->getRandomGenerator()->normal(t, mean, stddev); }, Type::float32);
}

Ptr<NodeInitializer> glorotUniform(bool fanIn, bool fanOut, float scalingFactor) {
  return fromLambda([fanIn, fanOut, scalingFactor](Tensor t) {
    float scale = sqrtf(6.0f / (t->shape()[-2] + t->shape()[-1]));
    if(fanIn && !fanOut)
      scale = sqrtf(3.0f / t->shape()[-2]); // results in columns of matrix to be ~unit length
    if(!fanIn && fanOut)
      scale = sqrtf(3.0f / t->shape()[-1]);

    scale *= scalingFactor;

    t->getBackend()->getRandomGenerator()->uniform(t, -scale, scale);
  }, Type::float32);
}

Ptr<NodeInitializer> glorotNormal(bool fanIn, bool fanOut, float scalingFactor) {
  return fromLambda([fanIn, fanOut, scalingFactor](Tensor t) {
    float scale = sqrtf(2.0f / (t->shape()[-2] + t->shape()[-1]));
    if(fanIn && !fanOut)
      scale = sqrtf(1.0f / t->shape()[-2]);
    if(!fanIn && fanOut)
      scale = sqrtf(1.0f / t->shape()[-1]);

    scale *= scalingFactor;

    t->getBackend()->getRandomGenerator()->normal(t, 0.f, scale);
  }, Type::float32);
}

Ptr<NodeInitializer> bernoulli(float prob, float scale, float shift) {
  return fromLambda([prob, scale, shift](Tensor t) { Bernoulli(t, prob, scale, shift); }, Type::float32);
}

Ptr<NodeInitializer> dropout(float dropProb) {
  return fromLambda([dropProb](Tensor t) { Dropout(t, dropProb); }, Type::float32);
}

// gumbel noise:
// -log(-log(uniform(0.f + eps, 1.f - eps)));
Ptr<NodeInitializer> gumbel(float eps) {
  return fromLambda([eps](Tensor tensor) {
    tensor->getBackend()->getRandomGenerator()->uniform(tensor, 0.f + eps, 1.f - eps);
    using namespace functional;
    Element(_1 = -log(-log(_1)), tensor);
  }, Type::float32);
}

template <typename T>
Ptr<NodeInitializer> fromVector(const std::vector<T>& v) {
  return fromLambda([v](Tensor t) { t->set(v.data(), v.data() + v.size()); }, typeId<T>());
}

template <typename T>
Ptr<NodeInitializer> fromVector(std::vector<T>&& v) {
  return fromLambda([v](Tensor t) { t->set(v.data(), v.data() + v.size()); }, typeId<T>());
}

template Ptr<NodeInitializer> fromVector<float16>(const std::vector<float16>& v);
template Ptr<NodeInitializer> fromVector<float>(const std::vector<float>& v);
template Ptr<NodeInitializer> fromVector<IndexType>(const std::vector<IndexType>& v);

// @TODO: can we remove the const& ones above? They always make a copy anyways, and often from a temp
template Ptr<NodeInitializer> fromVector<float16>  (std::vector<float16>  && v);
template Ptr<NodeInitializer> fromVector<float>    (std::vector<float>    && v);
template Ptr<NodeInitializer> fromVector<IndexType>(std::vector<IndexType>&& v);

Ptr<NodeInitializer> fromSparseVector(std::pair<std::vector<size_t>, std::vector<float>>& v) {
  return fromLambda([v](Tensor t) { t->set(1e-6); t->setSparse(v.first, v.second); });
}

// move this somewhere else
Ptr<NodeInitializer> fromWord2vec(const std::string& file,
                              int dimVoc,
                              int dimEmb,
                              bool normalize /*= false*/) {
  return fromLambda([file, dimVoc, dimEmb, normalize](Tensor t) {
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
  });
}

Ptr<NodeInitializer> fromItem(const io::Item& item) {
  if(item.mapped) {
    return fromLambda([item](Tensor tensor) {
      // @TODO: implement other types, for now croak loudly.
      ABORT_IF(tensor->getBackend()->getDeviceId().type != DeviceType::cpu,
               "Memory mapping only works for CPU tensors");
      ABORT_IF(tensor->type() != item.type,
               "Tensor type ({}) and type for mapping ({}) do not match",
               tensor->type(),
               item.type);
      ABORT_IF(tensor->shape() != item.shape,
               "Tensor shape ({}) and shape of mapped item ({}) do not match",
               tensor->shape(),
               item.shape);
      auto mp = MemoryPiece::New((uint8_t*)item.ptr, item.size()); // @TODO: this is not properly aligned now
      tensor->reset(mp);
    });
  } else {
    return fromLambda(
      [item](Tensor tensor) { tensor->set(item); },
      item.type);
  }
}

Ptr<NodeInitializer> fromTensor(Tensor externalTensor) {
  return fromLambda([externalTensor](Tensor t) { t->copyFrom(externalTensor); }, externalTensor->type());
}

// Computes Google's sinusoidal position embeddings
Ptr<NodeInitializer> sinusoidalPositionEmbeddings(int start) {
  return fromLambda([start](Tensor t) {
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

    t->set(vPos);
  }, Type::float32);
}

}  // namespace inits

}  // namespace marian
