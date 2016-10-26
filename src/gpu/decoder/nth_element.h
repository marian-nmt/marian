#pragma once

#include <vector>
#include <algorithm>

#include <cuda_runtime.h>

namespace GPU {


class NthElement {
  public:
    NthElement(size_t maxBeamSize, cudaStream_t& stream);

    void getNBestList(float* d_in, size_t N, size_t n,
                      std::vector<unsigned>& outKeys,
                      std::vector<float>& outValues);

    /* cudaFree(d_in); */
    /* cudaFree(d_out); */

    /* free(h_in); */
    /* free(h_out); */

  private:
    const int BLOCK_SIZE = 512;
    cudaStream_t& stream_;
    int *d_ind;

    float *d_out;

    int   *d_res_idx;
    float *d_res;

    int   *h_res_idx;
    float *h_res;
};


}  // namespace GPU
