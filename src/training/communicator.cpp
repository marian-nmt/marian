#include "training/communicator.h"

namespace marian {

// Compile this if cuda is not being compiled.
// Version with CUDA and/or NCCL is compiled in communicator.cu
#ifndef CUDA_FOUND
Ptr<Communicator> createCommunicator(
    const std::vector<Ptr<ExpressionGraph>>& graphs,
    bool /*noNccl*/) {
  return New<DefaultCommunicator>(graphs);
}
#endif

}  // namespace marian
