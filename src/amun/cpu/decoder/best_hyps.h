#pragma once

#include <vector>
#include <boost/iterator/permutation_iterator.hpp>

#include "common/scorer.h"
#include "common/god.h"
#include "common/exception.h"
#include "cpu/mblas/tensor.h"
#include "cpu/decoder/encoder_decoder.h"

namespace amunmt {
namespace CPU {

class BestHyps : public BaseBestHyps
{
  public:
    BestHyps(const God &god);

    void CalcBeam(
        const Beam& prevHyps,
        const std::vector<ScorerPtr>& scorers,
        const Words& filterIndices,
        std::vector<Beam>& beams,
        std::vector<unsigned>& beamSizes);

};

}  // namespace CPU
}  // namespace amunmt
