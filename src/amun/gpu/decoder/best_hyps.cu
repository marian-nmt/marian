#include "best_hyps.h"

using namespace std;

namespace amunmt {
namespace GPU {

BestHyps::BestHyps(const God &god)
      : BaseBestHyps(god),
        keys_(god.Get<unsigned>("beam-size") * god.Get<unsigned>("mini-batch")),
        costs_(god.Get<unsigned>("beam-size") * god.Get<unsigned>("mini-batch")),
        maxBeamSize_(god.Get<unsigned>("beam-size"))
{
  if (!god_.UseFusedSoftmax()) {
    NthElement *obj = new NthElement(god.Get<unsigned>("beam-size"), god.Get<unsigned>("mini-batch"));
    nthElement_.reset(obj);
  }
}

void BestHyps::DisAllowUNK(mblas::Tensor& Prob) {
  SetColumn(Prob, UNK_ID, std::numeric_limits<float>::lowest());
}

void BestHyps::FindBests(const std::vector<unsigned>& beamSizes, mblas::Tensor& Probs,
               std::vector<float>& outCosts,
               std::vector<unsigned>& outKeys,
               const bool isFirst)
{
  nthElement_->getNBestList(beamSizes, Probs, outCosts, outKeys, isFirst);
}

// fast fused softmax and nth_element
void BestHyps::FindBests(const std::vector<unsigned>& beamSizes, mblas::Tensor& Probs,
               mblas::Vector<NthOutBatch> &nBest,
               std::vector<float>& outCosts,
               std::vector<unsigned>& outKeys,
               const bool isFirst)
{
  getNBestList(beamSizes, Probs, nBest, outCosts, outKeys, isFirst);
}

std::vector<SoftAlignmentPtr> BestHyps::GetAlignments(const std::vector<ScorerPtr>& scorers,
                                            unsigned hypIndex)
{
  std::vector<SoftAlignmentPtr> alignments;
  for (auto& scorer : scorers) {
    if (GPU::EncoderDecoder* encdec = dynamic_cast<GPU::EncoderDecoder*>(scorer.get())) {
      const mblas::Tensor &attention = encdec->GetAttention();
      unsigned attLength = attention.dim(1);

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

// standard nth_element
void  BestHyps::CalcBeam(
    const Beam& prevHyps,
    const std::vector<ScorerPtr>& scorers,
    const Words& filterIndices,
    std::vector<Beam>& beams,
    std::vector<unsigned>& beamSizes)
{
  BEGIN_TIMER("CalcBeam");

  using namespace mblas;

  mblas::Tensor& Probs = static_cast<mblas::Tensor&>(scorers[0]->GetProbs());

  std::vector<float> vCosts;
  for (auto& h : prevHyps) {
    vCosts.push_back(h->GetCost());
  }

  mblas::copy(vCosts.data(),
              vCosts.size(),
              costs_.data(),
              cudaMemcpyHostToDevice);
  //mblas::copy(vCosts.begin(), vCosts.end(), costs_.begin());

  unsigned beamSizeSum = std::accumulate(beamSizes.begin(), beamSizes.end(), 0);

  std::vector<float> bestCosts;
  std::vector<unsigned> bestKeys;

  const bool isFirst = (vCosts[0] == 0.0f) ? true : false;

  if (god_.UseFusedSoftmax()) {
    const mblas::Tensor& b4 = *static_cast<const mblas::Tensor*>(scorers[0]->GetBias());
    mblas::Vector<NthOutBatch> &nBest = *static_cast<mblas::Vector<NthOutBatch>*>(scorers[0]->GetNBest());
    nBest.newSize(beamSizeSum);

    bool requireProb = maxBeamSize_ > 1 || god_.Get<bool>("n-best");
    //cerr << "doSoftmax=" << doSoftmax << endl;

    BEGIN_TIMER("GetProbs.LogSoftmaxAndNBest");
    mblas::LogSoftmaxAndNBest(nBest, Probs, b4, costs_, forbidUNK_, maxBeamSize_, beamSizes, beamSizeSum, isFirst, requireProb);
    PAUSE_TIMER("GetProbs.LogSoftmaxAndNBest");
    //std::cerr << "2Probs=" << Probs.Debug(1) << std::endl;

    FindBests(beamSizes, Probs, nBest, bestCosts, bestKeys, isFirst);
  }
  else {
    BroadcastVecColumn(weights_.at(scorers[0]->GetName()) * _1 + _2, Probs, costs_);

    for (unsigned i = 1; i < scorers.size(); ++i) {
      mblas::Tensor &currProbs = static_cast<mblas::Tensor&>(scorers[i]->GetProbs());

      Element(_1 + weights_.at(scorers[i]->GetName()) * _2, Probs, currProbs);
    }

    if (forbidUNK_) {
      DisAllowUNK(Probs);
    }

    FindBests(beamSizes, Probs, bestCosts, bestKeys, isFirst);
  }

  std::vector<std::vector<float>> breakDowns;
  if (god_.ReturnNBestList()) {
      breakDowns.push_back(bestCosts);
      for (unsigned i = 1; i < scorers.size(); ++i) {
        std::vector<float> modelCosts(beamSizeSum);
        mblas::Tensor &currProbs = static_cast<mblas::Tensor&>(scorers[i]->GetProbs());

        nthElement_->getValueByKey(modelCosts, currProbs);
        breakDowns.push_back(modelCosts);
      }
  }

  std::map<unsigned, unsigned> batchMap;
  unsigned tmp = 0;
  for (unsigned batchID = 0; batchID < beamSizes.size(); ++batchID) {
    for (unsigned t = 0; t < beamSizes[batchID]; ++t) {
      batchMap[tmp++] = batchID;
    }
  }

  for (unsigned i = 0; i < beamSizeSum; i++) {
    unsigned wordIndex = bestKeys[i] % Probs.dim(1);
    if (isInputFiltered_) {
      wordIndex = filterIndices[wordIndex];
    }

    unsigned hypIndex  = bestKeys[i] / Probs.dim(1);
    float cost = bestCosts[i];

    HypothesisPtr hyp;
    if (returnAttentionWeights_) {
      hyp.reset(new Hypothesis(prevHyps[hypIndex], wordIndex, hypIndex, cost,
                               GetAlignments(scorers, hypIndex)));
    } else {
      hyp.reset(new Hypothesis(prevHyps[hypIndex], wordIndex, hypIndex, cost));
    }

    //cerr << "god_.ReturnNBestList()=" << god_.ReturnNBestList() << endl;
    if(god_.ReturnNBestList()) {
      hyp->GetCostBreakdown().resize(scorers.size());
      float sum = 0;
      for (unsigned j = 0; j < scorers.size(); ++j) {
        if (j == 0)
          hyp->GetCostBreakdown()[0] = breakDowns[0][i];
        else {
          float cost = 0;
          if (j < scorers.size()) {
              if (prevHyps[hypIndex]->GetCostBreakdown().size() < scorers.size())
                const_cast<HypothesisPtr&>(prevHyps[hypIndex])->GetCostBreakdown().resize(scorers.size(), 0.0f);
              cost = breakDowns[j][i] + const_cast<HypothesisPtr&>(prevHyps[hypIndex])->GetCostBreakdown()[j];
          }
          sum += weights_.at(scorers[j]->GetName()) * cost;
          hyp->GetCostBreakdown()[j] = cost;
        }
      }
      hyp->GetCostBreakdown()[0] -= sum;
      hyp->GetCostBreakdown()[0] /= weights_.at(scorers[0]->GetName());
    }

    beams[batchMap[i]].push_back(hyp);
  }

  PAUSE_TIMER("CalcBeam");
}

//////////////////////////////////////////////////////////////////////////
void BestHyps::getNBestList(const std::vector<unsigned>& beamSizes,
                  mblas::Tensor& Probs,
                  mblas::Vector<NthOutBatch> &nBest,
                  std::vector<float>& outCosts,
                  std::vector<unsigned>& outKeys,
                  const bool isFirst) const
{
  GetPairs(nBest, outKeys, outCosts);
  assert(outCosts.size() == outKeys.size());

  /*
  cerr << "outCosts/outKeys=";
  for (unsigned i = 0; i < outKeys.size(); ++i) {
    cerr << "(" << outCosts[i] << "," << outKeys[i] << ") ";
  }
  cerr << endl;
  */
  //cerr << endl;
}

void BestHyps::GetPairs(mblas::Vector<NthOutBatch> &nBest,
              std::vector<unsigned>& outKeys,
              std::vector<float>& outValues) const
{
  //cerr << "top=" << top2.size() << " nBest=" << nBest.size() << endl;
  outKeys.resize(nBest.size());
  outValues.resize(nBest.size());

  std::vector<NthOutBatch> hostVec(nBest.size());
  mblas::copy(nBest.data(), nBest.size(), hostVec.data(), cudaMemcpyDeviceToHost);

  for (unsigned i = 0; i < nBest.size(); ++i) {
    outKeys[i] = hostVec[i].ind;
    outValues[i] = hostVec[i].score;
  }
}

} // namespace
}
