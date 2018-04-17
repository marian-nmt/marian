#pragma once

#include <functional>
#include <vector>
#include <map>

#include "common/types.h"
#include "scorer.h"

namespace amunmt {

class BaseBestHyps
{
  public:
    BaseBestHyps(const God &god);

    BaseBestHyps(const BaseBestHyps&) = delete;

    virtual void CalcBeam(
        const Beam& prevHyps,
        const std::vector<ScorerPtr>& scorers,
        const Words& filterIndices,
        std::vector<Beam>& beams,
        std::vector<unsigned>& beamSizes) = 0;

  protected:
    const God &god_;
    const bool forbidUNK_;
    const bool isInputFiltered_;
    const bool returnAttentionWeights_;
    const std::map<std::string, float> weights_;

};

typedef std::shared_ptr<BaseBestHyps> BaseBestHypsPtr;

}
