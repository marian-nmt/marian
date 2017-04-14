#pragma once

#include <vector>
#include <algorithm>
#include "matrix.h"
#include "array.h"


namespace amunmt {
namespace FPGA {

class NthElement {
public:
  NthElement() = delete;
  NthElement(const NthElement &copy) = delete;
  NthElement(const OpenCLInfo &openCLInfo, size_t maxBeamSize, size_t maxBatchSize);

  void getNBestList(const std::vector<size_t>& beamSizes, mblas::Matrix& Probs,
                    std::vector<float>& outCosts, std::vector<unsigned>& outKeys,
                    const bool isFirst);

protected:
  const OpenCLInfo &openCLInfo_;

  const int BLOCK_SIZE = 512;
  const int NUM_BLOCKS;

  Array<int> d_ind;
  Array<float> d_out;

  Array<int> d_batchPosition;
  Array<int> d_cumBeamSizes;

  void getNBestList(mblas::Matrix &probs, const std::vector<int>& batchFirstElementIdxs,
                                const std::vector<int>& cummulatedBeamSizes);

};

} // namespace
}
