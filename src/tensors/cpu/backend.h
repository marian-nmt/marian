#pragma once

#include <functional>
#include <random>

#include "common/config.h"
#include "tensors/backend.h"

namespace marian {
namespace cpu {

class Backend : public marian::Backend {
private:
  std::default_random_engine gen_;

public:
  Backend(DeviceId deviceId, size_t seed)
      : marian::Backend(deviceId, seed), gen_(seed_) {}

  void setDevice() {}

  void synchronize() {}

  std::default_random_engine& getRandomGenerator() { return gen_; }
};
}
}
