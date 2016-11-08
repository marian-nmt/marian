#pragma once

#include <vector>
#include <memory>

#ifdef CUDA
#include <thrust/host_vector.h>
using SoftAlignment = thrust::host_vector<float>;
#else
using SoftAlignment = std::vector<float>;
#endif

using SoftAlignmentPtr = std::shared_ptr<SoftAlignment>;
