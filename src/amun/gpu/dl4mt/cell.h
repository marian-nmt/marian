#pragma once
#include "gpu/mblas/matrix_functions.h"
#include "gpu/mblas/matrix_wrapper.h"
#include "gpu/mblas/handles.h"

namespace amunmt {
namespace GPU {

class Cell {
  public:
    virtual void GetNextState(mblas::Matrix& NextState,
                      const mblas::Matrix& State,
                      const mblas::Matrix& Context) const = 0;

    virtual size_t GetStateLength() const = 0;
};
}
}


