#include <vector>
#include "best_hyps.h"
#include "matrix.h"
#include "common/god.h"

using namespace std;

namespace amunmt {
namespace FPGA {

BestHyps::BestHyps(const God &god, const OpenCLInfo &openCLInfo)
: nthElement_(openCLInfo, god.Get<size_t>("beam-size"), god.Get<size_t>("mini-batch")),
  keys(openCLInfo, god.Get<size_t>("beam-size") * god.Get<size_t>("mini-batch")),
  Costs(openCLInfo, god.Get<size_t>("beam-size") * god.Get<size_t>("mini-batch")),
  weights_(god.GetScorerWeights())
{
  //std::cerr << "BestHyps::BestHyps" << std::endl;
}

void BestHyps::DisAllowUNK(mblas::Matrix& Prob)
{
  SetColumn(Prob, UNK_ID, std::numeric_limits<float>::lowest());
}

void BestHyps::FindBests(const std::vector<size_t>& beamSizes, mblas::Matrix& Probs,
               std::vector<float>& outCosts,
               std::vector<unsigned>& outKeys,
               const bool isFirst)
{
  nthElement_.getNBestList(beamSizes, Probs, outCosts, outKeys, isFirst);
}

void BestHyps::CalcBeam(
    const God &god,
    const Beam& prevHyps,
    const std::vector<ScorerPtr>& scorers,
    const Words& filterIndices,
    bool returnAlignment,
    std::vector<Beam>& beams,
    std::vector<size_t>& beamSizes
    )
{
  using namespace mblas;

  mblas::Matrix& Probs = static_cast<mblas::Matrix&>(scorers[0]->GetProbs());

  std::vector<float> vCosts;
  for (auto& h : prevHyps) {
    vCosts.push_back(h->GetCost());
  }

  Costs.Fill(vCosts);

  const bool isFirst = (vCosts[0] == 0.0f) ? true : false;

  float weight = weights_.at(scorers[0]->GetName());

  BroadcastVecColumnAddWeighted(Probs, weight, Costs);
  std::cerr << "1Probs=" << Probs.Debug(1) << std::endl;

  for (size_t i = 1; i < scorers.size(); ++i) {
    mblas::Matrix &currProbs = static_cast<mblas::Matrix&>(scorers[i]->GetProbs());

    float weight = weights_.at(scorers[0]->GetName());
    ElementAddWeighted(Probs, weight, currProbs);
  }
  std::cerr << "2Probs=" << Probs.Debug(1) << std::endl;

  if (!god.Get<bool>("allow-unk")) {
    DisAllowUNK(Probs);
  }

  size_t beamSizeSum = std::accumulate(beamSizes.begin(), beamSizes.end(), 0);

  std::vector<float> bestCosts;
  std::vector<unsigned> bestKeys;

  FindBests(beamSizes, Probs, bestCosts, bestKeys, isFirst);

}

}
}

