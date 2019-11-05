#include "tensors/rand.h"
#include "tensors/tensor.h"
#include "tensors/tensor_operators.h"

#ifdef CUDA_FOUND
#include "gpu/cuda_helpers.h"
#include <curand.h>
#endif

namespace marian {

class StdlibRandomGenerator : public RandomGenerator {
private:
  std::mt19937 engine_;

public:
  StdlibRandomGenerator(size_t seed)
  : RandomGenerator(seed), engine_((unsigned int)seed) {}

  virtual void uniform(Tensor tensor, float a, float b) override;
  virtual void normal(Tensor, float mean, float stddev) override;
};

#ifdef CUDA_FOUND
class CurandRandomGenerator : public RandomGenerator {
private:
  DeviceId deviceId_;
  curandGenerator_t generator_;

public:
  CurandRandomGenerator(size_t seed, DeviceId deviceId);
  ~CurandRandomGenerator();

  virtual void uniform(Tensor, float a, float b) override;
  virtual void normal(Tensor, float mean, float stddev) override;

};
#endif

void StdlibRandomGenerator::uniform(Tensor tensor, float a, float b) {
    matchOrAbort<float>(tensor->type());

    ABORT_IF(tensor->getBackend()->getDeviceId().type != DeviceType::cpu,
             "StdlibRandomGenerator can only be used for CPU tensors");

    auto dist = std::uniform_real_distribution<float>(a, b);
    auto gen = bind(dist, std::ref(engine_)); // does not change engine state without std::ref

    auto begin = tensor->data<float>();
    auto end   = tensor->data<float>() + tensor->size();
    std::generate(begin, end, gen);
}

void StdlibRandomGenerator::normal(Tensor tensor, float mean, float stddev) {
    matchOrAbort<float>(tensor->type());

    ABORT_IF(tensor->getBackend()->getDeviceId().type != DeviceType::cpu,
             "StdlibRandomGenerator can only be used for CPU tensors");

    auto dist = std::normal_distribution<float>(mean, stddev);
    auto gen = bind(dist, std::ref(engine_)); // does not change engine state without std::ref

    auto begin = tensor->data<float>();
    auto end   = tensor->data<float>() + tensor->size();
    std::generate(begin, end, gen);
}

#ifdef CUDA_FOUND

CurandRandomGenerator::CurandRandomGenerator(size_t seed, DeviceId deviceId)
: RandomGenerator(seed), deviceId_(deviceId) {
    if(deviceId_.type == DeviceType::gpu) {
      cudaSetDevice((int)deviceId_.no);
      CURAND_CHECK(curandCreateGenerator(&generator_, CURAND_RNG_PSEUDO_DEFAULT));
    }
    else {
      CURAND_CHECK(curandCreateGeneratorHost(&generator_, CURAND_RNG_PSEUDO_DEFAULT));
    }
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(generator_, seed_));
}

CurandRandomGenerator::~CurandRandomGenerator() {
  // No CUDA error checking as this is a destructor and we cannot do anything about errors anyway.
  if(deviceId_.type == DeviceType::gpu)
    cudaSetDevice((int)deviceId_.no);
  curandDestroyGenerator(generator_);
}

void CurandRandomGenerator::uniform(Tensor tensor, float a, float b) {
    matchOrAbort<float>(tensor->type());

    tensor->getBackend()->setDevice();
    CURAND_CHECK(curandGenerateUniform(generator_, tensor->data(), tensor->size()));

    // curandGenerateUniform has no range parameters (why?) so we need to
    // scale and shift inplace if range is different than [0, 1).
    using namespace functional;
    if(a != 0.f || b != 1.f)
        Element(_1 = (b - a) * _1 + a, tensor);
}

void CurandRandomGenerator::normal(Tensor tensor, float mean, float stddev) {
    matchOrAbort<float>(tensor->type());

    tensor->getBackend()->setDevice();
    CURAND_CHECK(curandGenerateNormal(generator_, tensor->data(), tensor->size(), mean, stddev));
}

#endif

Ptr<RandomGenerator> createRandomGenerator(size_t seed, DeviceId deviceId) {
#ifdef CUDA_FOUND
    return New<CurandRandomGenerator>(seed, deviceId);
#else
    ABORT_IF(deviceId.type != DeviceType::cpu,
             "StdlibRandomGenerator can only be used for CPU tensors");
    return New<StdlibRandomGenerator>(seed);
#endif
}

}