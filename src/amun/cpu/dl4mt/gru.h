#pragma once
#include "cpu/mblas/tensor.h"

namespace amunmt {
namespace CPU {
namespace dl4mt {

template <class Weights>
class GRU {
  public:
    GRU(const Weights& model)
    : w_(model) {
      using namespace mblas;
      WWx_ = Concat<byColumn, Tensor>(w_.W_, w_.Wx_);
      UUx_ = Concat<byColumn, Tensor>(w_.U_, w_.Ux_);
    }

    void GetNextState(mblas::Tensor& NextState,
                      const mblas::Tensor& State,
                      const mblas::Tensor& Context) const {
      RUH_ = Context * WWx_;
      if (w_.Gamma_1_.rows()) {
        LayerNormalization(RUH_, w_.Gamma_1_);
      }

      Temp_ = State * UUx_;
      if (w_.Gamma_2_.rows()) {
        LayerNormalization(Temp_, w_.Gamma_2_);
      }

      // @TODO: once broadcasting is available
      // implement this using blaze idioms
      ElementwiseOps(NextState, State);
    }

    void ElementwiseOps(mblas::Tensor& NextState,
                        const mblas::Tensor& State) const {

      using namespace mblas;
      using namespace blaze;

      const size_t rowNo = State.rows();
      const size_t colNo = State.columns();
      NextState.resize(rowNo, colNo);

      for(int j = 0; j < rowNo; ++j) {
        auto rowOut = row(NextState, j);
        auto rowState = row(State, j);

        auto rowRuh = row(RUH_, j);
        auto rowT   = row(Temp_, j);

        auto rowH   = subvector(rowRuh, 2 * colNo, colNo);
        auto rowT2  = subvector(rowT, 2 * colNo, colNo);

        for(int i = 0; i < colNo; ++i) {
          float ev1 = expapprox(-(rowRuh[i] + w_.B_(0, i) + rowT[i]));
          float r = 1.0 / (1.0 + ev1);

          int k = i + colNo;
          float ev2 = expapprox(-(rowRuh[k] + w_.B_(0, k) + rowT[k]));
          float u = 1.0 / (1.0 + ev2);

          float hv = rowH[i] + w_.Bx1_(0, i);
          float t2v = rowT2[i] + w_.Bx2_(0, i);
          hv = tanhapprox(hv + r * t2v);
          rowOut[i] = (1.0 - u) * hv + u * rowState[i];
        }
      }

    }

    size_t GetStateLength() const {
      return w_.U_.rows();
    }


  private:
    // Model matrices
    const Weights& w_;
    mutable mblas::Tensor WWx_;
    mutable mblas::Tensor UUx_;

    // reused to avoid allocation
    mutable mblas::Tensor RUH_;
    mutable mblas::Tensor Temp_;
};

}
}
}
