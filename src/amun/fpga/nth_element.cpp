#include "nth_element.h"

using namespace std;

namespace amunmt {
namespace FPGA {

NthElement::NthElement(const OpenCLInfo &openCLInfo, size_t maxBeamSize, size_t maxBatchSize)
:openCLInfo_(openCLInfo)
,NUM_BLOCKS(std::min(500, int(maxBeamSize * 85000 / (2 * BLOCK_SIZE)) + int(maxBeamSize * 85000 % (2 * BLOCK_SIZE) != 0)))
,d_batchPosition(openCLInfo, maxBatchSize + 1)
,d_cumBeamSizes(openCLInfo, maxBatchSize + 1)
,d_ind(openCLInfo, maxBatchSize * NUM_BLOCKS)
,d_out(openCLInfo, maxBatchSize * NUM_BLOCKS)
,maxBeamSize_(maxBeamSize)
,maxBatchSize_(maxBatchSize)
{

}

void NthElement::getNBestList(mblas::Matrix &probs, const std::vector<int>& batchFirstElementIdxs,
                              const std::vector<int>& cummulatedBeamSizes)
{
  d_batchPosition.Fill(batchFirstElementIdxs);
  d_cumBeamSizes.Fill(cummulatedBeamSizes);

  const int numBatches = batchFirstElementIdxs.size() - 1;

  mblas::MaxElement(d_out, d_ind, probs, numBatches, d_batchPosition);

}

void NthElement::getNBestList(const std::vector<size_t>& beamSizes, mblas::Matrix& Probs,
                  std::vector<float>& outCosts, std::vector<unsigned>& outKeys,
                  const bool isFirst)
{
  /*
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
  */

  //d_out.Fill(666);
  //d_ind.Fill(666);

  mblas::NthElement(d_out, d_ind, Probs, maxBeamSize_, maxBatchSize_);
  cerr << "d_out=" << d_out.Debug(2) << endl;
  cerr << "d_ind=" << d_ind.Debug(2) << endl;

}

}  // namespace
}

