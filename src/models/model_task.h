#pragma once

#include "models/amun.h"
#include "models/s2s.h"
#include "models/multi_s2s.h"
#include "models/hardatt.h"

#include "data/corpus.h"

namespace marian {
  
  struct ModelTask {
    virtual void run() = 0;
  };
  
  template <template <class> class TaskName,
            template <class, class> class Wrapper>
  Ptr<ModelTask> WrapModelType(Ptr<Config> options) {
    auto type = options->get<std::string>("type");
    
    if(type == "amun")
      return New<TaskName<Wrapper<Amun, data::Corpus>>>(options);
    else if(type == "s2s")
      return New<TaskName<Wrapper<S2S, data::Corpus>>>(options);
    else if(type == "multi-s2s")
      return New<TaskName<Wrapper<MultiS2S, data::Corpus>>>(options);
    else if(type == "hard-att")
      return New<TaskName<Wrapper<HardAtt, data::Corpus>>>(options);
    else if(type == "hard-soft-att")
      return New<TaskName<Wrapper<HardSoftAtt, data::Corpus>>>(options);
    else if(type == "multi-hard-att")
      return New<TaskName<Wrapper<MultiHardSoftAtt, data::Corpus>>>(options);
    else
      UTIL_THROW2("Unknown model type: " << type);
  }

}
