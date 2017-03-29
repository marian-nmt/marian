#pragma once

namespace amunmt {
namespace FPGA {

template <class Weights>
class GRU {
public:
  GRU(const OpenCLInfo &openCLInfo, const Weights& model)
  : w_(model)
  , WWx_(openCLInfo)
  , UUx_(openCLInfo)
  , RUH_(openCLInfo)
  , Temp_(openCLInfo)
  {

  }

  size_t GetStateLength() const {
    return w_.U_.dim(0);
  }

  void GetNextState(mblas::Matrix& NextState,
                    const mblas::Matrix& State,
                    const mblas::Matrix& Context) const
  {
    using namespace mblas;

    Prod(RUH_, Context, WWx_);

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

