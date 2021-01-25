#pragma once

#include <functional>
#include <random>

#include "common/config.h"
#include "tensors/backend.h"

namespace marian {
namespace cpu {

class Backend : public marian::Backend {
public:
  Backend(DeviceId deviceId, size_t seed) : marian::Backend(deviceId, seed) {}
  void setDevice() override {}
  void synchronize() override {}
};

}  // namespace cpu
}  // namespace marian
