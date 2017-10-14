#pragma once

namespace marian {

struct ModelTask {
  virtual void run() = 0;
};

struct ModelServiceTask {
  virtual void init() = 0;
  virtual std::vector<std::string> run(const std::vector<std::string>&) = 0;
};
}
