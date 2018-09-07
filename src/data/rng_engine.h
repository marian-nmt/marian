#pragma once

#include <random>
#include <sstream>

#include "common/config.h"

namespace marian {
namespace data {

/**
 * @brief Class providing an engine for pseudo-random number generation.
 */
class RNGEngine {
protected:
  std::mt19937 eng_;

public:
  RNGEngine() : eng_((unsigned int)Config::seed) {}
  RNGEngine(size_t eng) : eng_((unsigned int)eng) {}

  std::string getRNGState() {
    std::ostringstream oss;
    oss << eng_;
    return oss.str();
  }

  void setRNGState(std::string engineState) {
    std::istringstream iss(engineState);
    iss >> eng_;
  }
};
}  // namespace data
}  // namespace marian
