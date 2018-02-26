#pragma once
#include "gpu/mblas/matrix_functions.h"
#include "gpu/mblas/matrix_wrapper.h"
#include "gpu/mblas/handles.h"

namespace amunmt {
namespace GPU {

struct CellState {
  CellState(){
    output = std::unique_ptr<mblas::Tensor>(new mblas::Tensor());
    cell = std::unique_ptr<mblas::Tensor>(new mblas::Tensor());
  };

  CellState(std::unique_ptr<mblas::Tensor> cell, std::unique_ptr<mblas::Tensor> output):
    cell(std::move(cell)), output(std::move(output)) {}

  std::unique_ptr<mblas::Tensor> output;
  std::unique_ptr<mblas::Tensor> cell;
};
}
}


