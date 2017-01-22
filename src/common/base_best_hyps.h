#pragma once

#include <functional>
#include <vector>

#include "common/types.h"
#include "scorer.h"


class BestHypsBase
{
public:
  BestHypsBase() {}
  BestHypsBase(const BestHypsBase&) = delete;

  virtual void operator()(const God &god,
		std::vector<Beam>& beams,
        const Beam& prevHyps,
        std::vector<size_t>& beamSizes,
        const std::vector<ScorerPtr>& scorers,
        const Words& filterIndices,
        bool returnAlignment) = 0;

};

