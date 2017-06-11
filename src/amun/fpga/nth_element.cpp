#include "nth_element.h"
#include "common/utils.h"

using namespace std;

namespace amunmt {
namespace FPGA {

NthElement::NthElement(const OpenCLInfo &openCLInfo, size_t maxBeamSize, size_t maxBatchSize)
:openCLInfo_(openCLInfo)
,NUM_BLOCKS(std::min(500, int(maxBeamSize * 85000 / (2 * BLOCK_SIZE)) + int(maxBeamSize * 85000 % (2 * BLOCK_SIZE) != 0)))
,d_ind(openCLInfo, maxBatchSize * NUM_BLOCKS)
,d_out(openCLInfo, maxBatchSize * NUM_BLOCKS)
,maxBeamSize_(maxBeamSize)
,maxBatchSize_(maxBatchSize)
{

}

void NthElement::getNBestList(const std::vector<uint>& beamSizes, mblas::Matrix& Probs,
                  std::vector<float>& outCosts, std::vector<unsigned>& outKeys,
                  const bool isFirst)
{
  const OpenCLInfo &openCLInfo = Probs.GetOpenCLInfo();

  assert(beamSizes.size());
  size_t totalBeamSize = beamSizes[0];
  for (size_t i = 1; i < beamSizes.size(); ++i) {
    totalBeamSize += beamSizes[i];
  }
  //cerr << "totalBeamSize=" << totalBeamSize << endl;

  std::vector<uint> cummulatedBeamSizes(beamSizes.size() + 1);
  std::vector<uint> batchFirstElementIdxs(beamSizes.size() + 1);
  cummulatedBeamSizes[0] = 0;
  batchFirstElementIdxs[0] = 0;

  const size_t vocabSize = Probs.dim(1);
  for (size_t i = 0; i < beamSizes.size(); ++i) {

    cummulatedBeamSizes[i + 1] = cummulatedBeamSizes[i] + beamSizes[i];
    batchFirstElementIdxs[i + 1] = ((isFirst) ? (i + 1) : cummulatedBeamSizes[i + 1]) * vocabSize;
  }

  Array<uint> d_cummulatedBeamSizes(openCLInfo, cummulatedBeamSizes);
  Array<uint> d_batchFirstElementIdxs(openCLInfo, batchFirstElementIdxs);

  //cerr << "Probs=" << Probs.Debug() << endl;
  //assert(Probs.dim(0) == totalBeamSize);

  //cerr << "beamSizes=" << Debug(beamSizes, 2) << endl;
  //cerr << "beamSize=" << beamSize << endl;
  //cerr << "maxBeamSize_=" << maxBeamSize_ << endl;
  //cerr << "maxBatchSize_=" << maxBatchSize_ << endl;

  // create device vector of beamSizes
  vector<uint> beamSizesUint(beamSizes.size());
  std::copy(beamSizes.begin(), beamSizes.end(), beamSizesUint.begin());
  Array<uint> d_beamSizesUint(openCLInfo, beamSizesUint);

  //cerr << endl;
  //cerr << "Probs=" << Probs.Debug(1) << endl;
  //cerr << "d_beamSizesUint=" << d_beamSizesUint.Debug(2) << endl;
  //cerr << "maxBatchSize_=" << maxBatchSize_ << endl;
  //cerr << "d_cummulatedBeamSizes=" << d_cummulatedBeamSizes.Debug(2) << endl;
  //cerr << "d_batchFirstElementIdxs=" << d_batchFirstElementIdxs.Debug(2) << endl;
  mblas::NthElement(d_out, d_ind, Probs, d_beamSizesUint, maxBatchSize_, d_cummulatedBeamSizes, d_batchFirstElementIdxs);
  //cerr << "d_out=" << d_out.Debug(1) << endl;
  //cerr << "d_ind=" << d_ind.Debug(1) << endl;

  outCosts.resize(totalBeamSize);
  outKeys.resize(totalBeamSize);
  d_out.Get(outCosts.data(), outCosts.size());
  d_ind.Get(outKeys.data(), outKeys.size());

  //cerr << "outCosts=" << Debug(outCosts, 2) << endl;
  //cerr << "outKeys=" << Debug(outKeys, 2) << endl;


}

}  // namespace
}

