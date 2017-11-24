#include "types-gpu.h"
#include "mblas/handles.h"

namespace amunmt {
namespace GPU {

void HandleError(cudaError_t err, const char *file, int line ) {
  if (err != cudaSuccess) {
    std::cerr << "ERROR: " << cudaGetErrorString(err) << " in " << file << " at line " << line << std::endl;
    exit( EXIT_FAILURE );
  }
}


}
}
