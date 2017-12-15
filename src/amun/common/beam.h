#pragma once
#include <vector>
#include <map>
#include <unordered_map>
#include <string>
#include "hypothesis.h"
#include "sentences.h"

namespace amunmt {

typedef std::vector<HypothesisPtr> Beam;

////////////////////////////////////////////////////////////////////////////////////////////////////////

typedef std::vector<Beam> Beams;

}

