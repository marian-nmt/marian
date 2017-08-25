#pragma once
#include <map>
#include "common/base_best_hyps.h"
#include "array.h"
#include "nth_element.h"

namespace amunmt {
namespace FPGA {

class BestHyps : public BestHypsBase
{
public:
  BestHyps(const God &god, const OpenCLInfo &openCLInfo);

  void DisAllowUNK(mblas::Matrix& Prob);

  void FindBests(const std::vector<uint>& beamSizes, mblas::Matrix& Probs,
                 std::vector<float>& outCosts,
                 std::vector<unsigned>& outKeys,
                 const bool isFirst);

  virtual void CalcBeam(
      const Beam& prevHyps,
      const std::vector<ScorerPtr>& scorers,
      const Words& filterIndices,
      std::vector<Beam>& beams,
      std::vector<uint>& beamSizes
      );

protected:
  NthElement nthElement_;

  Array<unsigned> keys;
  Array<float> Costs;

};

}
}

