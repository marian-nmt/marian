#pragma once
#include "../mblas/matrix_functions.h"
#include "../mblas/matrix_wrapper.h"
#include "../mblas/handles.h"

namespace amunmt {
namespace GPUHalf {

struct CellState {
  CellState(){
    output = std::unique_ptr<mblas::Matrix>(new mblas::Matrix());
    cell = std::unique_ptr<mblas::Matrix>(new mblas::Matrix());
  };

  CellState(std::unique_ptr<mblas::Matrix> cell, std::unique_ptr<mblas::Matrix> output):
    cell(std::move(cell)), output(std::move(output)) {}

  std::unique_ptr<mblas::Matrix> output;
  std::unique_ptr<mblas::Matrix> cell;
};
}
}


