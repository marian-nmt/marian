#pragma once
#include <vector>
#include <map>
#include <unordered_map>
#include <string>
#include "hypothesis.h"
#include "sentences.h"

namespace amunmt {

using Hypotheses = std::vector<HypothesisPtr>;

////////////////////////////////////////////////////////////////////////////////////////////////////////

using Beams = std::vector<Hypotheses>;

}

