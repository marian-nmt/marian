#pragma once

namespace amunmt {
namespace FPGA {

template <class Weights>
class GRU {
public:
  GRU(const cl_context &context, const Weights& model)
  : w_(model)
  , WWx_(context)
  , UUx_(context)
  , RUH_(context)
  , Temp_(context)
  {

  }

  size_t GetStateLength() const {
    return w_.U_.Rows();
  }

protected:
  // Model matrices
  const Weights& w_;

  // reused to avoid allocation
  mutable mblas::Matrix WWx_;
  mutable mblas::Matrix UUx_;

  mutable mblas::Matrix RUH_;
  mutable mblas::Matrix Temp_;

};

}
}

