#pragma once
#include <vector>
#include <map>
#include <unordered_map>
#include <string>
#include "hypothesis.h"
#include "sentences.h"

namespace amunmt {

typedef std::vector<HypothesisPtr> Beam;
typedef std::vector<Beam> Beams;

std::string Debug(const Beam &vec, unsigned verbosity = 1);
std::string Debug(const Beams &vec, unsigned verbosity = 1);

}

