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
                        mblas::Matrix& Probs,
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
    Histories& histories)
{
  BEGIN_TIMER("CalcBeam");
  using namespace mblas;

  mblas::Matrix& Probs = static_cast<mblas::Matrix&>(scorer.GetProbs());

  std::vector<float> vCosts;
  for (const HypothesisPtr &h : prevHyps) {
    if (h) {
      vCosts.push_back(h->GetCost());
    }
  }

  mblas::copy(vCosts.data(),
              vCosts.size(),
              costs_.data(),
              cudaMemcpyHostToDevice);
  //mblas::copy(vCosts.begin(), vCosts.end(), costs_.begin());

  size_t beamSizeSum = histories.Sum();;

  std::vector<float> bestCosts;
  std::vector<unsigned> bestKeys;

  if (god_.UseFusedSoftmax()) {
    const mblas::Matrix& b4 = *static_cast<const mblas::Matrix*>(scorer.GetBias());
    mblas::Vector<NthOutBatch> &nBest = *static_cast<mblas::Vector<NthOutBatch>*>(scorer.GetNBest());
    nBest.newSize(beamSizeSum);

    BEGIN_TIMER("GetProbs.LogSoftmaxAndNBest");
    mblas::LogSoftmaxAndNBest(nBest, Probs, b4, costs_, forbidUNK_, maxBeamSize_, histories, beamSizeSum);
    PAUSE_TIMER("GetProbs.LogSoftmaxAndNBest");
    //std::cerr << "2Probs=" << Probs.Debug(1) << std::endl;

    FindBests(histories, Probs, nBest, bestCosts, bestKeys);
  }
  else {
    BroadcastVecColumn(weights_.at(scorer.GetName()) * _1 + _2, Probs, costs_);

    if (forbidUNK_) {
      DisAllowUNK(Probs);
    }

    FindBests(histories, Probs, bestCosts, bestKeys);
  }

  std::vector<std::vector<float>> breakDowns;
  if (god_.ReturnNBestList()) {
      breakDowns.push_back(bestCosts);
  }

  std::map<size_t, size_t> batchMap;
  size_t tmp = 0;
  for (size_t batchID = 0; batchID < histories.size(); ++batchID) {
    for (size_t t = 0; t < histories.GetBeamSize(batchID); ++t) {
      batchMap[tmp++] = batchID;
    }
  }

  for (size_t i = 0; i < beamSizeSum; i++) {
    size_t wordIndex = bestKeys[i] % Probs.dim(1);
    if (isInputFiltered_) {
      wordIndex = filterIndices[wordIndex];
    }

    size_t hypIndex  = bestKeys[i] / Probs.dim(1);
    float cost = bestCosts[i];

    HypothesisPtr hyp;
    if (returnAttentionWeights_) {
      hyp.reset(new Hypothesis(prevHyps[hypIndex], wordIndex, hypIndex, cost,
                               GetAlignments(scorer, hypIndex)));
    } else {
      hyp.reset(new Hypothesis(prevHyps[hypIndex], wordIndex, hypIndex, cost));
    }

    //cerr << "god_.ReturnNBestList()=" << god_.ReturnNBestList() << endl;
    if(god_.ReturnNBestList()) {
      hyp->GetCostBreakdown().resize(1);
      float sum = 0;
      hyp->GetCostBreakdown()[0] = breakDowns[0][i];
      hyp->GetCostBreakdown()[0] -= sum;
      hyp->GetCostBreakdown()[0] /= weights_.at(scorer.GetName());
    }

    size_t batchInd = batchMap[i];
    //cerr << "batchInd=" << batchInd << endl;
    HistoriesElementPtr &ele = histories.Get(batchInd);
    assert(ele);
    ele->GetHypotheses().push_back(hyp);
  }

  PAUSE_TIMER("CalcBeam");
}

} // namespace
}
