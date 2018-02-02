#include <iostream>
#include "types-gpu.h"
#include "mblas/handles.h"

using namespace std;

namespace amunmt {
namespace GPU {

void HandleError(cudaError_t err, const char *file, int line ) {
  if (err != cudaSuccess) {
    std::cerr << "ERROR: " << cudaGetErrorString(err) << " in " << file << " at line " << line << std::endl;
    abort();
    //exit( EXIT_FAILURE );
  }
}

void HandleErrorCublas(cublasStatus_t err, const char *file, int line ) {
  if (err != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "CUBLAS ERROR: " << err << " in " << file << " at line " << line << std::endl;
    abort();
    //exit( EXIT_FAILURE );
  }
}

}
}
