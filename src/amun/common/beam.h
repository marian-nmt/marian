#pragma once
#include <vector>
#include <map>
#include <unordered_map>
#include <string>
#include "hypothesis.h"
#include "sentences.h"

using namespace std;

namespace amunmt {

typedef std::vector<HypothesisPtr> Beam;
typedef std::vector<Beam> Beams;

std::string Debug(const Beam &vec, size_t verbosity = 1);
std::string Debug(const Beams &vec, size_t verbosity = 1);

}

