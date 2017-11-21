#pragma once

#include <functional>
#include <vector>
#include <map>

#include "common/types.h"
#include "scorer.h"

namespace amunmt {

class God;

class BestHypsBase
{
  public:
    BestHypsBase(
        const God &god,
        bool forbidUNK,
        bool isInputFiltered,
        bool returnAttentionWeights,
        const std::map<std::string, float>& weights)
    : god_(god),
      forbidUNK_(forbidUNK),
      isInputFiltered_(isInputFiltered),
      returnAttentionWeights_(returnAttentionWeights),
      weights_(weights)
    {}

    BestHypsBase(const BestHypsBase&) = delete;

    virtual void CalcBeam(
        const Beam& prevHyps,
        const std::vector<ScorerPtr>& scorers,
        const Words& filterIndices,
        std::vector<Beam>& beams,
        std::vector<uint>& beamSizes) = 0;

  protected:
    const God &god_;
    const bool forbidUNK_;
    const bool isInputFiltered_;
    const bool returnAttentionWeights_;
    const std::map<std::string, float> weights_;

};

typedef std::shared_ptr<BestHypsBase> BestHypsBasePtr;

}
