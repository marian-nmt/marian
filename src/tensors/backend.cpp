#include "tensors/backend.h"

#ifdef CUDA_FOUND
#include "tensors/gpu/backend.h"
#endif

#include "tensors/cpu/backend.h"

namespace marian {

Ptr<Backend> BackendByDeviceId(DeviceId deviceId, size_t seed) {
#ifdef CUDA_FOUND
  if(deviceId.type == DeviceType::gpu)
    return New<gpu::Backend>(deviceId, seed);
  else
#endif
    return New<cpu::Backend>(deviceId, seed);
}
}  // namespace marian
