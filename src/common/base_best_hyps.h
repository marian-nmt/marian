#pragma once

#include <functional>
#include <vector>

#include "common/types.h"
#include "scorer.h"


using BestHypsType = std::function<void(Beam&, const Beam&, const size_t,
                    const std::vector<ScorerPtr>&, const Words&, bool)>;
