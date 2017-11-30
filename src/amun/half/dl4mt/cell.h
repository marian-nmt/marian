#pragma once
#include "../mblas/matrix_functions.h"
#include "../mblas/matrix_wrapper.h"
#include "../mblas/handles.h"
#include "cellstate.h"

namespace amunmt {
namespace GPUHalf {

struct CellLength {
  CellLength() = default;
  CellLength(size_t cell, size_t output): cell(cell), output(output){}
  size_t output;
  size_t cell;
};

class Cell {
  public:
    virtual void GetNextState(CellState& NextState,
                      const CellState& State,
                      const mblas::Matrix& Context) const = 0;

    virtual CellLength GetStateLength() const = 0;
};
}
}


