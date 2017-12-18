#pragma once

#include <functional>
#include <vector>
#include <map>
#include <memory>

#include "common/types.h"

namespace amunmt {

class God;
class BeamSize;
class Scorer;

class Hypothesis;
using HypothesisPtr = std::shared_ptr<Hypothesis>;
using Hypotheses = std::vector<HypothesisPtr>;
using HypothesesBatch = std::vector<Hypotheses>;

class BestHypsBase
{
  public:
    BestHypsBase(const God &god);
    BestHypsBase(const BestHypsBase&) = delete;
    virtual ~BestHypsBase();

    virtual void  CalcBeam(
        const Hypotheses& prevHyps,
        Scorer &scorer,
        const Words& filterIndices,
        HypothesesBatch& beams,
        BeamSize& beamSizes) = 0;

  protected:
    const God &god_;
    const bool forbidUNK_;
    const bool isInputFiltered_;
    const bool returnAttentionWeights_;
    const std::map<std::string, float> weights_;

};

typedef std::shared_ptr<BestHypsBase> BestHypsBasePtr;

}
