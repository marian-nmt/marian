#pragma once

namespace marian {

struct ModelTask {
  virtual void run() = 0;
};

struct ModelServiceTask {
  virtual void init() = 0;
  virtual std::vector<std::string> run(const std::vector<std::string>&) = 0;
};

template <template <class> class TaskName, class Wrapper>
Ptr<ModelTask> WrapModelType(Ptr<Config> options) {
  return New<TaskName<Wrapper>>(options);
}

}
