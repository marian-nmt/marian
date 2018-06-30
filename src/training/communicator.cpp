#include "training/communicator.h"

namespace marian {

// Compile this if cuda is not being compiled in or
// if cuda is being compiled in, but NCCL is not available.
// Version with NCCL is compiled in communicator.cu
#if !defined(CUDA_FOUND) || (defined(CUDA_FOUND) && !defined(USE_NCCL))
Ptr<Communicator> createCommunicator(const std::vector<Ptr<ExpressionGraph>>& graphs) {
  return New<DefaultCommunicator>(graphs);
}
#endif

}
