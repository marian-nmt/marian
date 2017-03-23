#pragma once

#include <functional>
#include <vector>

#include "common/types.h"
#include "scorer.h"

namespace amunmt {

class BestHypsBase
{
public:
  BestHypsBase() {}
  BestHypsBase(const BestHypsBase&) = delete;

  virtual void CalcBeam(
      const God &god,
      const Beam& prevHyps,
      const std::vector<ScorerPtr>& scorers,
      const Words& filterIndices,
      bool returnAlignment,
      std::vector<Beam>& beams,
      std::vector<size_t>& beamSizes
      ) = 0;

};

typedef std::shared_ptr<BestHypsBase> BestHypsBasePtr;

}
