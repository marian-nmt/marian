#include <iostream>
#include "common/utils.h"
#include "tensor_wrapper.h"
#include "vector_wrapper.h"
#include "nth_element.h"
#include "tensor_functions.h"

using namespace std;

namespace amunmt {
namespace GPU {

NthElement::NthElement(unsigned maxBeamSize, unsigned maxBatchSize)
: d_breakdown(maxBeamSize)
, maxBeamSize_(maxBeamSize)
, maxBatchSize_(maxBatchSize)
{
  //cerr << "maxBatchSize=" << maxBatchSize << " maxBeamSize=" << maxBeamSize << endl;

  d_batchPosition.reserve(maxBatchSize + 1);
  d_cumBeamSizes.reserve(maxBatchSize + 1);

  d_res.reserve(maxBatchSize * maxBeamSize);
  h_res.reserve(maxBatchSize * maxBeamSize);
}

NthElement::~NthElement()
{
  //cerr << "FOO2" << endl;
}

void NthElement::getNBestList(const std::vector<unsigned>& beamSizes, mblas::Tensor& Probs,
                  std::vector<float>& outCosts, std::vector<unsigned>& outKeys,
                  const bool isFirst) {
  /*
  cerr << "beamSizes=" << beamSizes.size() << endl;
  cerr << Debug(beamSizes, 2) << endl;
  cerr << "Probs=" << Probs.Debug(0) << endl;
  cerr << "outCosts=" << outCosts.size() << endl;
  cerr << "outKeys=" << outKeys.size() << endl;
  cerr << "isFirst=" << isFirst << endl;
  cerr << endl;
  */
  std::vector<unsigned> cummulatedBeamSizes(beamSizes.size() + 1);
  std::vector<unsigned> batchFirstElementIdxs(beamSizes.size() + 1);
  cummulatedBeamSizes[0] = 0;
  batchFirstElementIdxs[0] = 0;

  const unsigned vocabSize = Probs.dim(1);
  for (unsigned i = 0; i < beamSizes.size(); ++i) {

    cummulatedBeamSizes[i + 1] = cummulatedBeamSizes[i] + beamSizes[i];
    batchFirstElementIdxs[i + 1] = ((isFirst) ? (i + 1) : cummulatedBeamSizes[i + 1]) * vocabSize;
  }

  unsigned numHypos = cummulatedBeamSizes.back();
  d_res.newSize(numHypos);
  h_res.resize(numHypos);

  //cerr << endl;
  //cerr << "numHypos=" << numHypos << endl;
  //cerr << "beamSizes=" << Debug(beamSizes, 2) << endl;
  //cerr << "cummulatedBeamSizes=" << Debug(cummulatedBeamSizes, 2) << endl;
  //cerr << "batchFirstElementIdxs=" << Debug(batchFirstElementIdxs, 2) << endl;
  //cerr << "1Probs=" << Probs.Debug() << endl;

  getNBestList(Probs, batchFirstElementIdxs, cummulatedBeamSizes);

  //cerr << "2Probs=" << Probs.Debug() << endl;
  //cerr << "cummulatedBeamSizes.back()=" << cummulatedBeamSizes.back() << endl;
  //cerr << "cummulatedBeamSizes=" << Debug(cummulatedBeamSizes, 2) << endl;
  GetPairs(numHypos, outKeys, outCosts);

  //cerr << "outCosts=" << Debug(outCosts, 2) << endl;
  //cerr << "outKeys=" << Debug(outKeys, 2) << endl;
}

void NthElement::getNBestList(mblas::Tensor &probs,
                              const std::vector<unsigned>& batchFirstElementIdxs,
                              const std::vector<unsigned>& cummulatedBeamSizes)
{
  const unsigned vocabSize = probs.dim(1);
  const unsigned numBlocks = unsigned(maxBeamSize_ * vocabSize / (2 * BLOCK_SIZE)) + unsigned(maxBeamSize_ * vocabSize % (2 * BLOCK_SIZE) != 0);
  const unsigned numBatches = batchFirstElementIdxs.size() - 1;

  d_out.newSize(maxBatchSize_ * numBlocks);

  //cerr << "cummulatedBeamSizes=" << cummulatedBeamSizes.size() << endl;
  d_batchPosition.newSize(batchFirstElementIdxs.size());
  d_cumBeamSizes.newSize(cummulatedBeamSizes.size());
  assert(d_batchPosition.size() == d_cumBeamSizes.size());

  mblas::copy(batchFirstElementIdxs.data(),
              batchFirstElementIdxs.size(),
              d_batchPosition.data(),
              cudaMemcpyHostToDevice);
  mblas::copy(cummulatedBeamSizes.data(),
              cummulatedBeamSizes.size(),
              d_cumBeamSizes.data(),
              cudaMemcpyHostToDevice);

  mblas::VectorWrapper<NthOut> outWrap(d_out);
  mblas::TensorWrapper<float> probsWrap(probs);
  mblas::VectorWrapper<unsigned> batchPositionWrap(d_batchPosition);
  mblas::VectorWrapper<NthOut> resWrap(d_res);
  mblas::VectorWrapper<unsigned> cumBeamSizesWrap(d_cumBeamSizes);

  gMaxElement<<<numBlocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float), mblas::CudaStreamHandler::GetStream()>>>
    (outWrap, probsWrap, batchPositionWrap, numBatches);
  HANDLE_ERROR(cudaGetLastError());

  gMaxElementUpdate<<<numBatches, BLOCK_SIZE, BLOCK_SIZE * sizeof(float), mblas::CudaStreamHandler::GetStream()>>>
    (outWrap,
     probsWrap,
     resWrap,
     batchPositionWrap,
     cumBeamSizesWrap,
     numBlocks);
  HANDLE_ERROR(cudaGetLastError());

  /*
  cerr << "numBlocks=" << numBlocks << endl;
  cerr << "numBatches=" << numBatches << endl;
  cerr << "threads=" << BLOCK_SIZE << endl;

  cerr << "outWrap=" << outWrap.Debug() << endl;

  cerr << "probsWrap=" << probsWrap.Debug() << endl;

  cerr << "batchPositionWrap=" << batchPositionWrap.Debug() << endl;
  cerr << mblas::Debug(d_batchPosition, 2) << endl;

  cerr << "resWrap=" << resWrap.Debug() << endl;
  cerr << mblas::Debug(d_res, 2) << endl;

  cerr << "cumBeamSizesWrap=" << cumBeamSizesWrap.Debug() << endl;
  //cerr << mblas::Debug(d_cumBeamSizes, 2) << endl;

  cerr << endl;
  */
}

void NthElement::GetPairs(unsigned number,
                    std::vector<unsigned>& outKeys,
                    std::vector<float>& outValues)
{
  mblas::copy(d_res.data(), d_res.size(), h_res.data(), cudaMemcpyDeviceToHost);
  HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()) );

  for (unsigned i = 0; i < number; ++i) {
    outKeys.push_back(h_res[i].ind);
    outValues.push_back(h_res[i].score);
  }
}

void NthElement::getValueByKey(std::vector<float>& out, const mblas::Tensor &d_in) const
{
  // need a model with multiple scorers to test this method
  assert(false);

  out.resize(d_breakdown.size());

  //mblas::VectorWrapper<float> breakdownWrap(d_breakdown);
  //const mblas::TensorWrapper<float> inWrap(d_in);
  //gGetValueByKey<<<1, lastN_, 0, stream_>>>
  //  (breakdownWrap, inWrap, h_res_idx, lastN_);
  /*
  cerr << "out="
      << out.size() << " "
      << d_breakdown.size() << " "
      << h_res.size()
      << endl;
  */
  mblas::copy(d_breakdown.data(), d_breakdown.size(), out.data(), cudaMemcpyDeviceToHost);
  HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
}

//////////////////////////////////////////////////////////////////////////

}  // namespace GPU
} // namespace amunmt
