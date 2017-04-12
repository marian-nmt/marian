#include <vector>
#include "best_hyps.h"
#include "matrix.h"
#include "common/god.h"

using namespace std;

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

  cerr << "CalcBeam0" << endl;

  mblas::Matrix& Probs = static_cast<mblas::Matrix&>(scorers[0]->GetProbs());
  cerr << "Probs=" << Probs.Debug(1) << endl;

  cerr << "CalcBeam1" << endl;
  std::vector<float> vCosts;
  for (auto& h : prevHyps) {
    vCosts.push_back(h->GetCost());
  }
  cerr << "CalcBeam2" << endl;

  Costs.Fill(vCosts);
  cerr << "CalcBeam3" << endl;

  const bool isFirst = (vCosts[0] == 0.0f) ? true : false;

  float weight = weights_.at(scorers[0]->GetName());
  cerr << "CalcBeam4" << endl;

  BroadcastVecColumnAddWeighted(weight, Probs, Costs);
  std::cerr << "Probs=" << Probs.Debug(1) << std::endl;

}

}
}

