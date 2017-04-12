#include <vector>
#include "best_hyps.h"
#include "matrix.h"
#include "common/god.h"

namespace amunmt {
namespace FPGA {

BestHyps::BestHyps(const God &god, const OpenCLInfo &openCLInfo)
: keys(openCLInfo, god.Get<size_t>("beam-size") * god.Get<size_t>("mini-batch")),
  Costs(openCLInfo, god.Get<size_t>("beam-size") * god.Get<size_t>("mini-batch")),
  weights_(god.GetScorerWeights())
{
  //std::cerr << "BestHyps::BestHyps" << std::endl;
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
  BroadcastVecColumnAddWeighted(weight, Probs, Costs);

}

}
}

