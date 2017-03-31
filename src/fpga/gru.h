#pragma once

namespace amunmt {
namespace FPGA {

template <class Weights>
class GRU {
public:
  GRU(const OpenCLInfo &openCLInfo, const Weights& model)
  : openCLInfo_(openCLInfo)
  , w_(model)
  , WWx_(openCLInfo)
  , UUx_(openCLInfo)
  , RUH_(openCLInfo)
  , Temp_(openCLInfo)
  {
    using namespace mblas;

    Transpose(WWx_, w_.W_);
    std::cerr << std::endl;
    std::cerr << "w_.W_=" << w_.W_.Debug(1) << std::endl;
    std::cerr << "1WWx_=" << WWx_.Debug(1) << std::endl;

    Matrix WxT(openCLInfo);
    Transpose(WxT, w_.Wx_);
    std::cerr << "w_.Wx_=" << w_.Wx_.Debug(1) << std::endl;
    std::cerr << "WxT=" << WxT.Debug(1) << std::endl;

    Concat(WWx_, WxT);
    std::cerr << "2WWx_=" << WWx_.Debug(1) << std::endl;

    Transpose(WWx_);
    std::cerr << "3WWx_=" << WWx_.Debug(1) << std::endl;

  }

  size_t GetStateLength() const {
    return w_.U_.dim(0);
  }

  void GetNextState(mblas::Matrix& NextState,
                    const mblas::Matrix& State,
                    const mblas::Matrix& Context) const
  {
    using namespace mblas;

    std::cerr << std::endl;
    std::cerr << "1RUH_=" << RUH_.Debug(1) << std::endl;
    std::cerr << "Context=" << Context.Debug(1) << std::endl;
    std::cerr << "WWx_" << WWx_.Debug(1) << std::endl;

    Prod(RUH_, Context, WWx_);

    std::cerr << "2RUH_=" << RUH_.Debug(1) << std::endl;

    if (w_.Gamma_1_) {
      Normalization(RUH_, RUH_, w_.Gamma_1_, 1e-9);
    }
    std::cerr << "3RUH_=" << RUH_.Debug(1) << std::endl;
    std::cerr << "State=" << State.Debug(1) << std::endl;
    std::cerr << "UUx_" << UUx_.Debug(1) << std::endl;

    Prod(Temp_, State, UUx_);
    std::cerr << "Temp_=" << Temp_.Debug(1) << std::endl;

  }

protected:
  // Model matrices
  const OpenCLInfo &openCLInfo_;
  const Weights& w_;

  // reused to avoid allocation
  mutable mblas::Matrix WWx_;
  mutable mblas::Matrix UUx_;

  mutable mblas::Matrix RUH_;
  mutable mblas::Matrix Temp_;

};

}
}

