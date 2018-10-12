#pragma once

#include <functional>
#include <random>

#include "common/config.h"
#include "tensors/backend.h"

namespace marian {
namespace cpu {

class Backend : public marian::Backend {
private:
  std::default_random_engine engine_;

public:
  Backend(DeviceId deviceId, size_t seed)
      : marian::Backend(deviceId, seed), engine_((unsigned int)seed_) {}

  void setDevice() override {}

  void synchronize() override {}

  std::default_random_engine& getRandomGenerator() { return engine_; }
};
}  // namespace cpu
}  // namespace marian
