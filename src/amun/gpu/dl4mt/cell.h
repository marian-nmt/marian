#pragma once
#include "gpu/mblas/tensor_functions.h"
#include "gpu/mblas/tensor_wrapper.h"
#include "gpu/mblas/handles.h"
#include "cellstate.h"

namespace amunmt {
namespace GPU {

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
                      const mblas::Tensor& Context) const = 0;

    virtual CellLength GetStateLength() const = 0;

    virtual std::string Debug(unsigned verbosity = 1) const = 0;
};
}
}


