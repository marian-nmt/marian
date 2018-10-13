#include "tensors/rand.h"
#include "tensors/tensor.h"
#include "tensors/tensor_operators.h"

#ifdef CUDA_FOUND
#include "gpu/cuda_helpers.h"

#include <curand.h>
#endif

namespace marian {

Ptr<RandomGenerator> randomGeneratorFactory(size_t seed, DeviceId deviceId) {
#ifdef CUDA_FOUND
    if(deviceId.type == DeviceType::gpu)
        return New<CurandRandomGenerator>(seed, deviceId);
    else
        return New<CurandRandomGenerator>(seed, deviceId);
        //return New<StdlibRandomGenerator>(seed);
#else
    ABORT_IF(deviceId.type != DeviceType::cpu,
             "StdlibRandomGenerator can only be used for CPU tensors");
    return New<StdlibRandomGenerator>(seed);
#endif
}

void StdlibRandomGenerator::uniform(Tensor tensor, float a, float b) {
    ABORT_IF(tensor->getBackend()->getDeviceId().type != DeviceType::cpu,
             "StdlibRandomGenerator can only be used for CPU tensors");

    std::uniform_real_distribution<float> dist(a, b);
    auto gen = std::bind(dist, engine_);

    auto begin = tensor->data<float>();
    auto end   = tensor->data<float>() + tensor->size();
    std::generate(begin, end, gen);
}

void StdlibRandomGenerator::normal(Tensor tensor, float mean, float stddev) {
    ABORT_IF(tensor->getBackend()->getDeviceId().type != DeviceType::cpu,
             "StdlibRandomGenerator can only be used for CPU tensors");

    std::normal_distribution<float> dist(mean, stddev);
    auto gen = std::bind(dist, engine_);

    auto begin = tensor->data<float>();
    auto end   = tensor->data<float>() + tensor->size();
    std::generate(begin, end, gen);
}

#ifdef CUDA_FOUND

CurandRandomGenerator::CurandRandomGenerator(size_t seed, DeviceId deviceId)
: RandomGenerator(seed), deviceId_(deviceId) {
    CURAND_CHECK(curandCreateGeneratorHost(&cpuGenerator, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(cpuGenerator, seed_));

    if(deviceId_.type == DeviceType::gpu) {
      cudaSetDevice(deviceId_.no);
      CURAND_CHECK(curandCreateGenerator(&gpuGenerator, CURAND_RNG_PSEUDO_DEFAULT));
      CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gpuGenerator, seed_));
    }
}

CurandRandomGenerator::~CurandRandomGenerator() {
    CURAND_CHECK(curandDestroyGenerator(cpuGenerator));
    if(deviceId_.type == DeviceType::gpu) {
        cudaSetDevice(deviceId_.no);
        CURAND_CHECK(curandDestroyGenerator(gpuGenerator));
    }
}

void CurandRandomGenerator::uniform(Tensor tensor, float a, float b) {
    tensor->getBackend()->setDevice();

    if(tensor->getBackend()->getDeviceId().type == DeviceType::gpu)
        CURAND_CHECK(curandGenerateUniform(gpuGenerator, tensor->data(), tensor->size()));
    else
        CURAND_CHECK(curandGenerateUniform(cpuGenerator, tensor->data(), tensor->size()));

    // curandGenerateUniform has no range parameters (why?) so we need to
    // scale and shift inplace if range is different than [0, 1).
    using namespace functional;
    if(a != 0.f || b != 1.f)
        Element(_1 = (b - a) * _1 + a, tensor);
}

void CurandRandomGenerator::normal(Tensor tensor, float mean, float stddev) {
    tensor->getBackend()->setDevice();

    if(tensor->getBackend()->getDeviceId().type == DeviceType::gpu)
        CURAND_CHECK(curandGenerateNormal(gpuGenerator, tensor->data(), tensor->size(), mean, stddev));
    else
        CURAND_CHECK(curandGenerateNormal(cpuGenerator, tensor->data(), tensor->size(), mean, stddev));
}

#endif

}