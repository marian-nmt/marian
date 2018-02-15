#include <algorithm>

#include "backend/dispatch.h"

namespace marian {
  namespace cpu {
    
    void Dropout(Ptr<Backend> backend, Tensor tensor, float p) {
      ABORT("Not implemented");
      std::fill(tensor->data(), tensor->data() + tensor->size(), p);
    }
    
  }
}
