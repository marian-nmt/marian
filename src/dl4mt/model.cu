#include "model.h"

std::string ENC_NAME(const std::string prefix, const std::string suffix, const size_t index) {
  return (index ? prefix + "_" + std::to_string(index) + suffix : prefix + suffix);
}
