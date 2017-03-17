#include "best_hyps.h"

namespace amunmt {
namespace FPGA {

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

}

}
}

