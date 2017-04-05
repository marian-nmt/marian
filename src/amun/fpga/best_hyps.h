#pragma once

#include "common/base_best_hyps.h"

namespace amunmt {
namespace FPGA {

class BestHyps : public BestHypsBase
{
public:
  virtual void CalcBeam(
      const God &god,
      const Beam& prevHyps,
      const std::vector<ScorerPtr>& scorers,
      const Words& filterIndices,
      bool returnAlignment,
      std::vector<Beam>& beams,
      std::vector<size_t>& beamSizes
      );

};

}
}

