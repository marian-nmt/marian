#include "tensors/backend.h"

#include "tensors/gpu/backend.h"
#include "tensors/cpu/backend.h"

namespace marian {

Ptr<Backend> BackendByDevice(DeviceId deviceId, size_t seed) {
  if(deviceId.type == DeviceType::gpu)
    return New<gpu::Backend>(deviceId, seed);
  else
    return New<cpu::Backend>(deviceId, seed);
}

}
