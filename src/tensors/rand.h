#pragma once

#include "common/definitions.h"

#include <random>

namespace marian {

class TensorBase;
typedef IPtr<TensorBase> Tensor;

class RandomGenerator {
protected:
  size_t seed_;

public:
  RandomGenerator(size_t seed) : seed_(seed) { }
  virtual ~RandomGenerator() {}
  virtual void uniform(Tensor, float a, float b) = 0;
  virtual void normal(Tensor, float mean, float stddev) = 0;
};

Ptr<RandomGenerator> createRandomGenerator(size_t /*seed*/, DeviceId);

}
