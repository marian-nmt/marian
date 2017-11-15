#include <numeric>
#include <vector>
#include "best_hyps.h"
#include "matrix.h"
#include "common/god.h"

using namespace std;

namespace amunmt {
namespace FPGA {

BestHyps::BestHyps(const God &god, const OpenCLInfo &openCLInfo)
: BestHypsBase(
    !god.Get<bool>("allow-unk"),
    god.Get<bool>("n-best"),
    god.Get<std::vector<std::string>>("softmax-filter").size(),
    god.Get<bool>("return-alignment") || god.Get<bool>("return-soft-alignment") || god.Get<bool>("return-nematus-alignment"),
    god.GetScorerWeights()),
  nthElement_(openCLInfo, god.Get<size_t>("beam-size"), god.Get<size_t>("mini-batch")),
  keys(openCLInfo, god.Get<size_t>("beam-size") * god.Get<size_t>("mini-batch")),
  Costs(openCLInfo, god.Get<size_t>("beam-size") * god.Get<size_t>("mini-batch"))
{
  //std::cerr << "BestHyps::BestHyps" << std::endl;
}

void BestHyps::DisAllowUNK(mblas::Matrix& Prob)
{
  SetColumn(Prob, UNK_ID, std::numeric_limits<float>::lowest());
}

void BestHyps::FindBests(const std::vector<uint>& beamSizes, mblas::Matrix& Probs,
               std::vector<float>& outCosts,
               std::vector<unsigned>& outKeys,
               const bool isFirst)
{
  nthElement_.getNBestList(beamSizes, Probs, outCosts, outKeys, isFirst);
}

void BestHyps::CalcBeam(
    const Beam& prevHyps,
    const std::vector<ScorerPtr>& scorers,
    const Words& filterIndices,
    std::vector<Beam>& beams,
    std::vector<uint>& beamSizes
    )
{
  /*
  using namespace mblas;

  mblas::Matrix& Probs = static_cast<mblas::Matrix&>(scorers[0]->GetProbs());
  //cerr << "Probs=" << Probs.Debug(1) << endl;

  std::vector<float> vCosts;
  for (auto& h : prevHyps) {
    vCosts.push_back(h->GetCost());
  }

  Costs.Set(vCosts);
  //cerr << "Costs=" << Costs.Debug(1) << endl;

  const bool isFirst = (vCosts[0] == 0.0f) ? true : false;

  float weight = weights_.at(scorers[0]->GetName());

  BroadcastVecColumnAddWeighted(Probs, weight, Costs);
  //std::cerr << "1Probs=" << Probs.Debug(1) << std::endl;

  for (size_t i = 1; i < scorers.size(); ++i) {
    mblas::Matrix &currProbs = static_cast<mblas::Matrix&>(scorers[i]->GetProbs());

    float weight = weights_.at(scorers[0]->GetName());
    ElementAddWeighted(Probs, weight, currProbs);
  }
  //std::cerr << "2Probs=" << Probs.Debug(1) << std::endl;

  if (!god.Get<bool>("allow-unk")) {
    DisAllowUNK(Probs);
  }

  size_t beamSizeSum = std::accumulate(beamSizes.begin(), beamSizes.end(), 0);

  std::vector<float> bestCosts;
  std::vector<unsigned> bestKeys;

  //std::cerr << "beamSizes=" << amunmt::Debug(beamSizes, 2) << " " << std::endl;
  //std::cerr << "isFirst=" << isFirst << " " << std::endl;

  FindBests(beamSizes, Probs, bestCosts, bestKeys, isFirst);
  //std::cerr << "bestCosts=" << amunmt::Debug(bestCosts, 2) << " " << std::endl;
  //std::cerr << "bestKeys=" << amunmt::Debug(bestKeys, 2) << std::endl;

  std::vector<std::vector<float>> breakDowns;
  bool doBreakdown = god.Get<bool>("n-best");
  if (doBreakdown) {
    // TODO
  }

  bool filter = god.Get<std::vector<std::string>>("softmax-filter").size();

  std::map<size_t, size_t> batchMap;
  size_t tmp = 0;
  for (size_t batchID = 0; batchID < beamSizes.size(); ++batchID) {
    for (size_t t = 0; t < beamSizes[batchID]; ++t) {
      batchMap[tmp++] = batchID;
    }
  }

  for (size_t i = 0; i < beamSizeSum; i++) {
    size_t wordIndex = bestKeys[i] % Probs.dim(1);
    if (filter) {
      wordIndex = filterIndices[wordIndex];
    }

    size_t hypIndex  = bestKeys[i] / Probs.dim(1);
    float cost = bestCosts[i];

    HypothesisPtr hyp;
    if (returnAlignment) {
      //hyp.reset(new Hypothesis(prevHyps[hypIndex], wordIndex, hypIndex, cost,
      //                         GetAlignments(scorers, hypIndex)));
    } else {
      hyp.reset(new Hypothesis(prevHyps[hypIndex], wordIndex, hypIndex, cost));
    }

    if(doBreakdown) {
      // TODO
    }

    beams[batchMap[i]].push_back(hyp);
  }
  */
}

}
}

