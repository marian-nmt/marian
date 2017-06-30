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

  void getNBestList(const std::vector<uint>& beamSizes, mblas::Matrix& Probs,
                    std::vector<float>& outCosts, std::vector<unsigned>& outKeys,
                    const bool isFirst);

protected:
  const OpenCLInfo &openCLInfo_;

  const int BLOCK_SIZE = 512;
  const int NUM_BLOCKS;

  Array<unsigned> d_ind;
  Array<float> d_out;

  size_t maxBeamSize_;
  size_t maxBatchSize_;

};

} // namespace
}
