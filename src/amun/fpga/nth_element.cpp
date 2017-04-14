#include "nth_element.h"

namespace amunmt {
namespace FPGA {

NthElement::NthElement(const OpenCLInfo &openCLInfo, size_t maxBeamSize, size_t maxBatchSize)
:openCLInfo_(openCLInfo)
{
  cl_int err;

  d_batchPosition = clCreateBuffer(openCLInfo_.context,  CL_MEM_READ_WRITE,  (maxBatchSize + 1) * sizeof(int), NULL, &err);
  CheckError(err);

  d_cumBeamSizes = clCreateBuffer(openCLInfo_.context,  CL_MEM_READ_WRITE,  (maxBatchSize + 1) * sizeof(int), NULL, &err);
  CheckError(err);

}

void NthElement::getNBestList(mblas::Matrix &probs, const std::vector<int>& batchFirstElementIdxs,
                              const std::vector<int>& cummulatedBeamSizes)
{

}

void NthElement::getNBestList(const std::vector<size_t>& beamSizes, mblas::Matrix& Probs,
                  std::vector<float>& outCosts, std::vector<unsigned>& outKeys,
                  const bool isFirst)
{
  std::vector<int> cummulatedBeamSizes(beamSizes.size() + 1);
  std::vector<int> batchFirstElementIdxs(beamSizes.size() + 1);
  cummulatedBeamSizes[0] = 0;
  batchFirstElementIdxs[0] = 0;

  const size_t vocabSize = Probs.dim(1);
  for (size_t i = 0; i < beamSizes.size(); ++i) {

    cummulatedBeamSizes[i + 1] = cummulatedBeamSizes[i] + beamSizes[i];
    batchFirstElementIdxs[i + 1] = ((isFirst) ? (i + 1) : cummulatedBeamSizes[i + 1]) * vocabSize;
  }

  getNBestList(Probs, batchFirstElementIdxs, cummulatedBeamSizes);

}

}  // namespace
}

