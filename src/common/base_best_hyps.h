#pragma once

#include <functional>
#include <vector>

#include "common/types.h"
#include "scorer.h"


using BestHypsType = std::function<void(std::vector<Beam>&, const Beam&,
                    std::vector<size_t>& beamSizes,
                    const std::vector<ScorerPtr>&, const Words&, bool)>;
