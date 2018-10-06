#pragma once

#include <string>

namespace marian {

struct ModelTask {
  virtual void run() = 0;
};

struct ModelServiceTask {
  virtual void init() = 0;
  virtual std::string run(const std::string&) = 0;
};
}  // namespace marian
