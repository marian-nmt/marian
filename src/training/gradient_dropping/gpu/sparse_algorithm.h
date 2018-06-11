#pragma once

#include "common/definitions.h"
#include "tensors/backend.h"
#include "tensors/tensor.h"

namespace marian {
namespace gpu {
/**
 * @brief Output[i] is lower_bound of values[i] in data.
 *
 * @return A vector of size values.size
 */
std::vector<int> lower_bounds(int* data,
                              std::vector<int> values,
                              int size,
                              DeviceId device);

int buildSparse(Tensor t, float* data, int* indices);

void scatterAdd(Tensor t, float* data, int* indices, int size, int offset);

void scatterUpdate(Tensor t, float* data, int* indices, int size, int offset);

void gather(Tensor t, float* data, int* indices, int size, int offset);
}
}
