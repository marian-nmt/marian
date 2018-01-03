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

void BestHyps::FindBests(const Histories& histories,
                          mblas::Matrix& Probs,
                          std::vector<float>& outCosts,
                          std::vector<unsigned>& outKeys)
{
  nthElement_->getNBestList(histories, Probs, outCosts, outKeys);
}

// fast fused softmax and nth_element
void BestHyps::FindBests(const Histories& histories,
                        const mblas::Matrix& Probs,
                        mblas::Vector<NthOutBatch> &nBest,
                        std::vector<float>& outCosts,
                        std::vector<unsigned>& outKeys)
{
  getNBestList(histories, Probs, nBest, outCosts, outKeys);
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
void BestHyps::getNBestList(const Histories& histories,
                  const mblas::Matrix& Probs,
                  mblas::Vector<NthOutBatch> &nBest,
                  std::vector<float>& outCosts,
                  std::vector<uint>& outKeys) const
{
  //cerr << "2outKeys=" << Debug(outKeys, 2) << endl;
  GetPairs(nBest, outKeys, outCosts);
  //cerr << "3outKeys=" << Debug(outKeys, 2) << endl;
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
  HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));

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
    Histories& histories)
{
  BEGIN_TIMER("CalcBeam");
  using namespace mblas;

  //cerr << "CalcBeam1" << endl;
  mblas::Matrix& Probs = static_cast<mblas::Matrix&>(scorer.GetProbs());
  //cerr << "Probs=" << Probs.Debug(0) << endl;
  //cerr << "prevHyps=" << prevHyps.size() << endl;
  //cerr << "2histories=" << histories.Debug() << endl;

  std::vector<float> vCosts;
  for (const HypothesisPtr &h : prevHyps) {
    assert(h);
    vCosts.push_back(h->GetCost());
  }
  //cerr << "CalcBeam2" << endl;

  mblas::copy(vCosts.data(),
              vCosts.size(),
              costs_.data(),
              cudaMemcpyHostToDevice);
  //mblas::copy(vCosts.begin(), vCosts.end(), costs_.begin());
  //cerr << "CalcBeam3" << endl;

  size_t numHypos = histories.GetTotalBeamSize();
  //cerr << "CalcBeam4" << endl;
  //cerr << "numHypos=" << numHypos << endl;

  std::vector<float> bestCosts;
  std::vector<unsigned> bestKeys;

  if (god_.UseFusedSoftmax()) {
    //cerr << "CalcBeam5" << endl;
    const mblas::Matrix& b4 = *static_cast<const mblas::Matrix*>(scorer.GetBias());
    mblas::Vector<NthOutBatch> &nBest = *static_cast<mblas::Vector<NthOutBatch>*>(scorer.GetNBest());
    nBest.newSize(numHypos);
    /*
    std::cerr << "Probs=" << Probs.Debug(0) << std::endl;
    std::cerr << "b4=" << b4.Debug(0) << std::endl;
    std::cerr << "costs_=" << costs_.Debug(0) << std::endl;
    std::cerr << "histories=" << histories.Debug(1) << std::endl;
    std::cerr << "forbidUNK_=" << forbidUNK_ << std::endl;
    std::cerr << "maxBeamSize_=" << maxBeamSize_ << std::endl;
    std::cerr << "numHypos=" << numHypos << std::endl;
    */
    BEGIN_TIMER("GetProbs.LogSoftmaxAndNBest");
    mblas::LogSoftmaxAndNBest(nBest, Probs, b4, costs_, histories, forbidUNK_, maxBeamSize_);
    PAUSE_TIMER("GetProbs.LogSoftmaxAndNBest");
    //cerr << "nBest=" << nBest.Debug(2) << endl;

    FindBests(histories, Probs, nBest, bestCosts, bestKeys);
  }
  else {
    BroadcastVecColumn(weights_.at(scorer.GetName()) * _1 + _2, Probs, costs_);

    if (forbidUNK_) {
      DisAllowUNK(Probs);
    }

    FindBests(histories, Probs, bestCosts, bestKeys);
  }
  //cerr << "CalcBeam6" << endl;
  //cerr << "bestKeys=" << Debug(bestKeys, 2) << endl;

  std::vector<std::vector<float>> breakDowns;
  if (god_.ReturnNBestList()) {
      breakDowns.push_back(bestCosts);
  }
  //cerr << "CalcBeam7" << endl;

  std::vector<unsigned> batchMap = histories.Hypo2Batch();
  //cerr << "CalcBeam8" << endl;
  //cerr << "batchMap=" << Debug(batchMap, 2) << endl;

  for (size_t i = 0; i < numHypos; i++) {
    //cerr << "CalcBeam9=" << i << endl;
    size_t wordIndex = bestKeys[i] % Probs.dim(1);
    if (isInputFiltered_) {
      wordIndex = filterIndices[wordIndex];
    }
    //cerr << "CalcBeam10=" << i << endl;

    size_t hypIndex  = bestKeys[i] / Probs.dim(1);
    float cost = bestCosts[i];
    //cerr << "CalcBeam11=" << i << endl;
    //cerr << "bestKeys[i]=" << bestKeys[i] << endl;
    //cerr << "hypIndex=" << hypIndex << endl;
    //cerr << "prevHyps=" << prevHyps.size() << endl;

    assert(hypIndex < prevHyps.size());
    const HypothesisPtr &prevHyp = prevHyps[hypIndex];
    HypothesisPtr hyp;
    if (returnAttentionWeights_) {
      hyp.reset(new Hypothesis(prevHyp, wordIndex, hypIndex, cost,
                               GetAlignments(scorer, hypIndex)));
    } else {
      hyp.reset(new Hypothesis(prevHyp, wordIndex, hypIndex, cost));
    }
    //cerr << "CalcBeam12=" << i << endl;

    //cerr << "god_.ReturnNBestList()=" << god_.ReturnNBestList() << endl;
    if(god_.ReturnNBestList()) {
      hyp->GetCostBreakdown().resize(1);
      float sum = 0;
      hyp->GetCostBreakdown()[0] = breakDowns[0][i];
      hyp->GetCostBreakdown()[0] -= sum;
      hyp->GetCostBreakdown()[0] /= weights_.at(scorer.GetName());
    }
    //cerr << "CalcBeam13=" << i << endl;

    size_t batchInd = batchMap[i];
    //cerr << "CalcBeam14=" << i << endl;
    //cerr << "batchInd=" << batchInd << endl;
    HistoriesElementPtr &ele = histories.Get(batchInd);
    //cerr << "CalcBeam15=" << i << endl;
    assert(ele);
    ele->GetHypotheses().push_back(hyp);
    //cerr << "CalcBeam16=" << i << endl;
  }
  //cerr << "CalcBeam17" << endl;

  PAUSE_TIMER("CalcBeam");
}

} // namespace
}
