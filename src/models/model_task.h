#pragma once

#include <string>

namespace marian {

struct ModelTask {
  virtual void run() = 0;
};

struct ModelServiceTask {
  virtual std::string run(const std::string&) = 0;
};
}  // namespace marian
