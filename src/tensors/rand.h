#pragma once

#include "common/definitions.h"

#include <random>

#ifdef CUDA_FOUND
struct curandGenerator_st;
typedef struct curandGenerator_st* curandGenerator_t;
#endif

namespace marian {

class TensorBase;
typedef Ptr<TensorBase> Tensor;

class RandomGenerator {
protected:
  size_t seed_;

public:
  RandomGenerator(size_t seed) : seed_(seed) { }

  virtual void uniform(Tensor, float a, float b) = 0;
  virtual void normal(Tensor, float mean, float stddev) = 0;
};

Ptr<RandomGenerator> randomGeneratorFactory(size_t /*seed*/, DeviceId);

class StdlibRandomGenerator : public RandomGenerator {
private:
  std::mt19937 engine_;

public:
  StdlibRandomGenerator(size_t seed) : RandomGenerator(seed) {}

  virtual void uniform(Tensor tensor, float a, float b) override;
  virtual void normal(Tensor, float mean, float stddev) override;
};

#ifdef CUDA_FOUND
class CurandRandomGenerator : public RandomGenerator {
private:
  DeviceId deviceId_;
  curandGenerator_t cpuGenerator;
  curandGenerator_t gpuGenerator;

public:
  CurandRandomGenerator(size_t seed, DeviceId deviceId);
  ~CurandRandomGenerator();

  virtual void uniform(Tensor tensor, float a, float b) override;
  virtual void normal(Tensor, float mean, float stddev) override;

};
#endif

}