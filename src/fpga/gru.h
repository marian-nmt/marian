#pragma once

namespace amunmt {
namespace FPGA {

template <class Weights>
class GRU {
public:
  GRU(const cl_context &context, const cl_device_id &device, const Weights& model)
  : w_(model)
  , WWx_(context, device)
  , UUx_(context, device)
  , RUH_(context, device)
  , Temp_(context, device)
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

