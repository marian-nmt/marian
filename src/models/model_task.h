#pragma once

#include "models/s2s.h"
#include "models/amun.h"
#include "models/lm.h"
#include "models/hardatt.h"
#include "models/multi_s2s.h"

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

template <template <class> class TaskName, template <class> class Wrapper>
Ptr<ModelTask> WrapModelType(Ptr<Config> options) {
  auto type = options->get<std::string>("type");

  REGISTER_MODEL("s2s", S2S);
  REGISTER_MODEL("amun", Amun);
  REGISTER_MODEL("hard-att", HardAtt);
  REGISTER_MODEL("hard-soft-att", HardSoftAtt);

  REGISTER_MODEL("multi-s2s", MultiS2S);
  REGISTER_MODEL("multi-hard-att", MultiHardSoftAtt);

  REGISTER_MODEL("lm", LM);

  UTIL_THROW2("Unknown model type: " << type);
}

}
