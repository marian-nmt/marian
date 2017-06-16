#pragma once

#include <functional>
#include <vector>
#include <map>

#include "common/types.h"
#include "scorer.h"

namespace amunmt {

class BestHypsBase
{
  public:
    BestHypsBase(
        bool forbidUNK,
        bool returnNBestList,
        bool isInputFiltered,
        bool returnAttentionWeights,
        const std::map<std::string, float>& weights)
    : forbidUNK_(forbidUNK),
      returnNBestList_(returnNBestList),
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
    const bool forbidUNK_;
    const bool returnNBestList_;
    const bool isInputFiltered_;
    const bool returnAttentionWeights_;
    const std::map<std::string, float> weights_;

};

typedef std::shared_ptr<BestHypsBase> BestHypsBasePtr;

}
