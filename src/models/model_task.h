#pragma once

#include "models/s2s.h"
//#include "models/amun.h"
//#include "models/lm.h"
//#include "models/hardatt.h"
//#include "models/multi_s2s.h"

namespace marian {

struct ModelTask {
  virtual void run() = 0;
};

struct ModelServiceTask {
  virtual void init() = 0;
  virtual std::vector<std::string> run(const std::vector<std::string>&) = 0;
};

#define REGISTER_MODEL(name, model) \
do { \
  if(type == name) \
    return New<TaskName<Wrapper<model>>>(options); \
} while(0)

template <template <class> class TaskName, class Wrapper>
Ptr<ModelTask> WrapModelType(Ptr<Config> options) {
  return New<TaskName<Wrapper>>(options);
}

}
