#pragma once

#include <string>

namespace marian {

// Returns list of non-advanced cache variables used by CMake
std::string cmakeBuildOptions();

// Returns list of advanced cache variables used by CMake
std::string cmakeBuildOptionsAdvanced();

} // namespace marian
