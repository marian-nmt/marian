#pragma once

#include <vector>
#include <algorithm>
#include "matrix.h"


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
  cl_mem d_batchPosition;
  cl_mem d_cumBeamSizes;

  void getNBestList(mblas::Matrix &probs, const std::vector<int>& batchFirstElementIdxs,
                                const std::vector<int>& cummulatedBeamSizes);

};

} // namespace
}
