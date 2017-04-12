#pragma once

#include "models/amun.h"
#include "models/s2s.h"
//#include "models/multi_s2s.h"

namespace marian {
  
  struct ModelTask {
    virtual void run() = 0;
  };
  
  template <template <class> class TaskName,
            template <class> class Wrapper>
  Ptr<ModelTask> WrapModelType(Ptr<Config> options) {
    auto type = options->get<std::string>("type");
    
    if(type == "amun")
      return New<TaskName<Wrapper<Amun>>>(options);
    else if(type == "s2s")
      return New<TaskName<Wrapper<S2S>>>(options);
    //else if(type == "multi-s2s")
    // return New<TaskName<Wrapper<MultiS2S>>>(options);
    else
      UTIL_THROW2("Unknown model type: " << type);
  }

}