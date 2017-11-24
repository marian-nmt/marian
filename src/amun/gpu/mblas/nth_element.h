#pragma once

#include <vector>
#include <algorithm>

#include <cuda.h>
#include "gpu/mblas/matrix.h"
#include "nth_element_kernels.h"

namespace amunmt {
namespace GPU {


class NthElement {
  public:
    NthElement() = delete;
    NthElement(const NthElement &copy) = delete;
    NthElement(uint maxBeamSize, uint maxBatchSize);
    virtual ~NthElement();

    // standard nth_element
    void getNBestList(const std::vector<uint>& beamSizes,
                      mblas::Matrix& Probs,
                      std::vector<float>& outCosts,
                      std::vector<uint>& outKeys,
                      const bool isFirst=false);

    void GetPairs(uint number,
                  std::vector<uint>& outKeys,
                  std::vector<float>& outValues);

    void getValueByKey(std::vector<float>& out, const mblas::Matrix &d_in) const;

  private:
    const uint BLOCK_SIZE = 512;

    mblas::TMatrix<NthOut> d_out;

    mblas::TMatrix<NthOut> d_res;
    std::vector<NthOut> h_res;

    mblas::TMatrix<float> d_breakdown;
    mblas::TMatrix<uint> d_batchPosition;
    mblas::TMatrix<uint> d_cumBeamSizes;

    uint maxBeamSize_, maxBatchSize_;

    void getNBestList(mblas::Matrix &probs,
                      const std::vector<uint>& batchFirstElementIdxs,
                      const std::vector<uint>& cummulatedBeamSizes);


};

}  // namespace GPU
}
