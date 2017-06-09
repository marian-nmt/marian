#pragma once

#include <vector>
#include <algorithm>

#include <cuda.h>
#include "gpu/mblas/matrix.h"

namespace amunmt {
namespace GPU {

class NthElement {
  public:
    NthElement() = delete;
    NthElement(const NthElement &copy) = delete;
    NthElement(size_t maxBeamSize, size_t maxBatchSize, cudaStream_t& stream);
    virtual ~NthElement();

    void getNBestList(mblas::Matrix &probs, const std::vector<int>& batchFirstElementIdxs,
                              const std::vector<int>& cummulatedBeamSizes);
    void getNBestList(const std::vector<size_t>& beamSizes, mblas::Matrix& Probs,
                      std::vector<float>& outCosts, std::vector<unsigned>& outKeys,
                      const bool isFirst=false);

    void GetPairs(size_t number,
                  std::vector<unsigned>& outKeys,
                  std::vector<float>& outValues);

    void getValueByKey(std::vector<float>& out, float* d_in);

  private:
    const int BLOCK_SIZE = 512;
    const int numBlocks_;
    cudaStream_t& stream_;

    DeviceVector<int> d_ind;
    DeviceVector<float> d_out;

    DeviceVector<int>   d_res_idx;
    DeviceVector<float> d_res;

    int   *h_res_idx;
    float *h_res;

    DeviceVector<float> d_breakdown;
    DeviceVector<int> d_batchPosition;
    DeviceVector<int> d_cumBeamSizes;
    size_t lastN;
};

}  // namespace GPU
}
