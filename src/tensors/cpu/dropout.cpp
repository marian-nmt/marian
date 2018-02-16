#include <algorithm>

#include "tensors/dispatch.h"

namespace marian {
  namespace cpu {

    void Dropout(Tensor tensor, float p) {
      ABORT("Not implemented");
      std::fill(tensor->data(), tensor->data() + tensor->size(), p);
    }

  }
}
