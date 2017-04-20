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

void NthElement::getNBestList(const std::vector<size_t>& beamSizes, mblas::Matrix& Probs,
                  std::vector<float>& outCosts, std::vector<unsigned>& outKeys,
                  const bool isFirst)
{
  assert(beamSizes.size() == 1);
  size_t beamSize = beamSizes[0];
  //cerr << "beamSize=" << beamSize << endl;
  //cerr << "maxBeamSize_=" << maxBeamSize_ << endl;
  //cerr << "maxBatchSize_=" << maxBatchSize_ << endl;

  //d_out.Fill(666);
  //d_ind.Fill(666);

  mblas::NthElement(d_out, d_ind, Probs, beamSize, maxBatchSize_);
  //cerr << "d_out=" << d_out.Debug(2) << endl;
  //cerr << "d_ind=" << d_ind.Debug(2) << endl;

  outCosts.resize(beamSize);
  outKeys.resize(beamSize);
  d_out.Get(outCosts.data(), outCosts.size());
  d_ind.Get(outKeys.data(), outKeys.size());

  //cerr << "outCosts=" << Debug(outCosts, 2) << endl;
  //cerr << "outKeys=" << Debug(outKeys, 2) << endl;


}

}  // namespace
}

