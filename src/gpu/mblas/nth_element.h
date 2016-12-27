#pragma once

#include <vector>
#include <algorithm>

#include <cuda.h>
#include "gpu/mblas/matrix.h"

namespace GPU {

class NthElement {
  public:
    NthElement(size_t maxBeamSize, size_t maxBatchSize, cudaStream_t& stream);

    void getNBestList(float* d_in, size_t N, size_t n, size_t pos=0);
    void getNBestList(const std::vector<size_t>& beamSizes, mblas::Matrix& Probs,
                      std::vector<float>& outCosts, std::vector<unsigned>& outKeys,
                      const bool isFirst=false);

    void GetPairs(size_t number,
                  std::vector<unsigned>& outKeys,
                  std::vector<float>& outValues);

    void getValueByKey(std::vector<float>& out, float* d_in);

  private:
    const int BLOCK_SIZE = 512;
    cudaStream_t& stream_;
    int *d_ind;

    float *d_out;

    int   *d_res_idx;
    float *d_res;

    int   *h_res_idx;
    float *h_res;

    float  *d_breakdown;
    size_t lastN;
};

}  // namespace GPU
