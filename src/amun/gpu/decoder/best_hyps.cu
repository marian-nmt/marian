#include "best_hyps.h"
#include "common/histories.h"
#include "common/hypothesis.h"

using namespace std;

namespace amunmt {
namespace GPU {

BestHyps::BestHyps(const God &god)
      : BestHypsBase(god),
        keys_(god.Get<size_t>("beam-size") * god.Get<size_t>("mini-batch")),
        costs_(god.Get<size_t>("beam-size") * god.Get<size_t>("mini-batch")),
        maxBeamSize_(god.Get<uint>("beam-size"))
{
  if (!god_.UseFusedSoftmax()) {
    NthElement *obj = new NthElement(god.Get<size_t>("beam-size"), god.Get<size_t>("mini-batch"));
    nthElement_.reset(obj);
  }
}

BestHyps::~BestHyps()
{
  //cerr << "~BestHyps" << endl;
}

void BestHyps::DisAllowUNK(mblas::Matrix& Prob) {
  SetColumn(Prob, UNK_ID, std::numeric_limits<float>::lowest());
}

void BestHyps::FindBests(const Histories& beamSizes,
                          mblas::Matrix& Probs,
                          std::vector<float>& outCosts,
                          std::vector<unsigned>& outKeys)
{
  nthElement_->getNBestList(beamSizes, Probs, outCosts, outKeys);
}

// fast fused softmax and nth_element
void BestHyps::FindBests(const Histories& beamSizes,
                        mblas::Matrix& Probs,
                        mblas::Vector<NthOutBatch> &nBest,
                        std::vector<float>& outCosts,
                        std::vector<unsigned>& outKeys)
{
  getNBestList(beamSizes, Probs, nBest, outCosts, outKeys);
}

std::vector<SoftAlignmentPtr> BestHyps::GetAlignments(const std::vector<ScorerPtr>& scorers,
                                            size_t hypIndex)
{
  std::vector<SoftAlignmentPtr> alignments;
  for (auto& scorer : scorers) {
    if (GPU::EncoderDecoder* encdec = dynamic_cast<GPU::EncoderDecoder*>(scorer.get())) {
      const mblas::Matrix &attention = encdec->GetAttention();
      size_t attLength = attention.dim(1);

      SoftAlignment *softAlignment = new SoftAlignment(attLength);
      mblas::copy(
          attention.data() + hypIndex * attLength,
          attLength,
          softAlignment->data(),
          cudaMemcpyDeviceToHost
      );

      alignments.emplace_back(softAlignment);
    } else {
      amunmt_UTIL_THROW2("Return Alignment is allowed only with Nematus scorer.");
    }
  }
  return alignments;
}

//////////////////////////////////////////////////////////////////////////
void BestHyps::getNBestList(const Histories& beamSizes,
                  mblas::Matrix& Probs,
                  mblas::Vector<NthOutBatch> &nBest,
                  std::vector<float>& outCosts,
                  std::vector<uint>& outKeys) const
{
  GetPairs(nBest, outKeys, outCosts);
  assert(outCosts.size() == outKeys.size());

  /*
  cerr << "outCosts/outKeys=";
  for (size_t i = 0; i < outKeys.size(); ++i) {
    cerr << "(" << outCosts[i] << "," << outKeys[i] << ") ";
  }
  cerr << endl;
  */
  //cerr << endl;
}

void BestHyps::GetPairs(mblas::Vector<NthOutBatch> &nBest,
              std::vector<uint>& outKeys,
              std::vector<float>& outValues) const
{
  //cerr << "top=" << top2.size() << " nBest=" << nBest.size() << endl;
  outKeys.resize(nBest.size());
  outValues.resize(nBest.size());

  std::vector<NthOutBatch> hostVec(nBest.size());
  mblas::copy(nBest.data(), nBest.size(), hostVec.data(), cudaMemcpyDeviceToHost);

  for (size_t i = 0; i < nBest.size(); ++i) {
    outKeys[i] = hostVec[i].ind;
    outValues[i] = hostVec[i].score;
  }
}

/////////////////////////////////////////////////////////////////////////////////////
// const-batch2
std::vector<SoftAlignmentPtr> BestHyps::GetAlignments(Scorer &scorer,
                                            size_t hypIndex)
{
  std::vector<SoftAlignmentPtr> alignments;
  GPU::EncoderDecoder &encdec = static_cast<GPU::EncoderDecoder&>(scorer);
  const mblas::Matrix &attention = encdec.GetAttention();
  size_t attLength = attention.dim(1);

  SoftAlignment *softAlignment = new SoftAlignment(attLength);
  mblas::copy(
      attention.data() + hypIndex * attLength,
      attLength,
      softAlignment->data(),
      cudaMemcpyDeviceToHost
  );

  alignments.emplace_back(softAlignment);

  return alignments;

}

// standard nth_element
void  BestHyps::CalcBeam(
    const Hypotheses& prevHyps,
    Scorer &scorer,
    const Words& filterIndices,
    HypothesesBatch& beams,
    Histories& beamSizes)
{
  BEGIN_TIMER("CalcBeam");
  using namespace mblas;
  HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
  cerr << "BestHyps::CalcBeam1" << endl;

  mblas::Matrix& Probs = static_cast<mblas::Matrix&>(scorer.GetProbs());

  std::vector<float> vCosts;
  for (const HypothesisPtr &h : prevHyps) {
    if (h) {
      vCosts.push_back(h->GetCost());
    }
  }
  HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
  cerr << "BestHyps::CalcBeam2" << endl;

  mblas::copy(vCosts.data(),
              vCosts.size(),
              costs_.data(),
              cudaMemcpyHostToDevice);
  //mblas::copy(vCosts.begin(), vCosts.end(), costs_.begin());
  HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
  cerr << "BestHyps::CalcBeam3" << endl;

  size_t beamSizeSum = beamSizes.Sum();;

  HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
  cerr << "BestHyps::CalcBeam4" << endl;

  std::vector<float> bestCosts;
  std::vector<unsigned> bestKeys;

  if (god_.UseFusedSoftmax()) {
    HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
    cerr << "BestHyps::CalcBeam5" << endl;
    const mblas::Matrix& b4 = *static_cast<const mblas::Matrix*>(scorer.GetBias());
    mblas::Vector<NthOutBatch> &nBest = *static_cast<mblas::Vector<NthOutBatch>*>(scorer.GetNBest());
    nBest.newSize(beamSizeSum);

    BEGIN_TIMER("GetProbs.LogSoftmaxAndNBest");
    mblas::LogSoftmaxAndNBest(nBest, Probs, b4, costs_, forbidUNK_, maxBeamSize_, beamSizes, beamSizeSum);
    PAUSE_TIMER("GetProbs.LogSoftmaxAndNBest");
    //std::cerr << "2Probs=" << Probs.Debug(1) << std::endl;

    FindBests(beamSizes, Probs, nBest, bestCosts, bestKeys);
    HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
    cerr << "BestHyps::CalcBeam6" << endl;
  }
  else {
    HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
    cerr << "BestHyps::CalcBeam7" << endl;
    BroadcastVecColumn(weights_.at(scorer.GetName()) * _1 + _2, Probs, costs_);

    if (forbidUNK_) {
      DisAllowUNK(Probs);
    }

    FindBests(beamSizes, Probs, bestCosts, bestKeys);
    HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
    cerr << "BestHyps::CalcBeam8" << endl;
  }

  HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
  cerr << "BestHyps::CalcBeam9" << endl;

  std::vector<std::vector<float>> breakDowns;
  if (god_.ReturnNBestList()) {
      breakDowns.push_back(bestCosts);
  }

  HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
  cerr << "BestHyps::CalcBeam10" << endl;

  std::map<size_t, size_t> batchMap;
  size_t tmp = 0;
  for (size_t batchID = 0; batchID < beamSizes.size(); ++batchID) {
    for (size_t t = 0; t < beamSizes.GetBeamSize(batchID); ++t) {
      batchMap[tmp++] = batchID;
    }
  }

  HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
  cerr << "BestHyps::CalcBeam11" << endl;

  for (size_t i = 0; i < beamSizeSum; i++) {
    HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
    cerr << "BestHyps::CalcBeam12" << endl;

    size_t wordIndex = bestKeys[i] % Probs.dim(1);
    if (isInputFiltered_) {
      wordIndex = filterIndices[wordIndex];
    }

    HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
    cerr << "BestHyps::CalcBeam12.1" << endl;

    size_t hypIndex  = bestKeys[i] / Probs.dim(1);
    float cost = bestCosts[i];

    HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
    cerr << "BestHyps::CalcBeam12.2" << endl;
    cerr << "prevHyps=" << prevHyps.size() << endl;
    cerr << "hypIndex=" << hypIndex << endl;

    HypothesisPtr hyp;
    if (returnAttentionWeights_) {
      hyp.reset(new Hypothesis(prevHyps[hypIndex], wordIndex, hypIndex, cost,
                               GetAlignments(scorer, hypIndex)));
    } else {
      hyp.reset(new Hypothesis(prevHyps[hypIndex], wordIndex, hypIndex, cost));
    }

    HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
    cerr << "BestHyps::CalcBeam12.3" << endl;

    //cerr << "god_.ReturnNBestList()=" << god_.ReturnNBestList() << endl;
    if(god_.ReturnNBestList()) {
      hyp->GetCostBreakdown().resize(1);
      float sum = 0;
      hyp->GetCostBreakdown()[0] = breakDowns[0][i];
      hyp->GetCostBreakdown()[0] -= sum;
      hyp->GetCostBreakdown()[0] /= weights_.at(scorer.GetName());
    }

    HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
    cerr << "BestHyps::CalcBeam12.4" << endl;

    beams[batchMap[i]].push_back(hyp);

    HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
    cerr << "BestHyps::CalcBeam12.5" << endl;

    HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
    cerr << "BestHyps::CalcBeam13" << endl;
  }

  HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
  cerr << "BestHyps::CalcBeam14" << endl;

  PAUSE_TIMER("CalcBeam");
}

} // namespace
}
