#pragma once
#include "cpu/mblas/matrix.h"
#include <iomanip>

namespace amunmt {
namespace CPU {

template <class Weights>
class GRU {
  public:
    GRU(const Weights& model)
      : w_(model),
        layerNormalization_(w_.W_lns_.rows())
    {
      if (!layerNormalization_) {
        WWx_ = mblas::Concat<mblas::byColumn, mblas::Matrix>(w_.W_, w_.Wx_);
        UUx_ = mblas::Concat<mblas::byColumn, mblas::Matrix>(w_.U_, w_.Ux_);
      }
    }

    void GetNextState(
      mblas::Matrix& nextState,
      const mblas::Matrix& state,
      const mblas::Matrix& context) const
    {
      // std::cerr << "Get next state" << std::endl;
      if (layerNormalization_) {
        RUH_1_ = context * w_.W_;
        mblas::AddBiasVector<mblas::byRow>(RUH_1_, w_.B_);
        LayerNormalization(RUH_1_, w_.W_lns_, w_.W_lnb_);

        RUH_2_ = context * w_.Wx_;
        mblas::AddBiasVector<mblas::byRow>(RUH_2_, w_.Bx1_);
        LayerNormalization(RUH_2_, w_.Wx_lns_, w_.Wx_lnb_);

        RUH_ = mblas::Concat<mblas::byColumn, mblas::Matrix>(RUH_1_, RUH_2_);

        Temp_1_ = state * w_.U_;
        mblas::AddBiasVector<mblas::byRow>(Temp_1_, w_.Bx3_);
        LayerNormalization(Temp_1_, w_.U_lns_, w_.U_lnb_);

        Temp_2_ = state * w_.Ux_;
        mblas::AddBiasVector<mblas::byRow>(Temp_2_, w_.Bx2_);
        LayerNormalization(Temp_2_, w_.Ux_lns_, w_.Ux_lnb_);

        Temp_ = mblas::Concat<mblas::byColumn, mblas::Matrix>(Temp_1_, Temp_2_);

        ElementwiseOpsLayerNorm(nextState, state);

      } else {
        RUH_ = context * WWx_;
        Temp_ = state * UUx_;
        ElementwiseOps(nextState, state);
      }
    }

    void ElementwiseOps(mblas::Matrix& NextState, const mblas::Matrix& State) const {
      using namespace mblas;
      using namespace blaze;

      const int rowNo = State.rows();
      const int colNo = State.columns();
      NextState.resize(rowNo, colNo);

      for (int j = 0; j < rowNo; ++j) {
        auto rowOut = row(NextState, j);
        auto rowState = row(State, j);

        auto rowRuh = row(RUH_, j);
        auto rowT   = row(Temp_, j);

        auto rowH   = subvector(rowRuh, 2 * colNo, colNo);
        auto rowT2  = subvector(rowT, 2 * colNo, colNo);

        for (int i = 0; i < colNo; ++i) {
          float ev1 = expapprox(-(rowRuh[i] + w_.B_(0, i) + rowT[i]));
          float r = 1.0f / (1.0f + ev1);

          int k = i + colNo;
          float ev2 = expapprox(-(rowRuh[k] + w_.B_(0, k) + rowT[k]));
          float u = 1.0f / (1.0f + ev2);

          float hv = rowH[i] + w_.Bx1_(0, i);
          float t2v = rowT2[i];
          hv = tanhapprox(hv + r * t2v);
          rowOut[i] = (1.0f - u) * hv + u * rowState[i];
        }
      }
    }

    void ElementwiseOpsLayerNorm(mblas::Matrix& NextState, const mblas::Matrix& State) const {
      using namespace mblas;
      using namespace blaze;

      const int rowNo = State.rows();
      const int colNo = State.columns();
      NextState.resize(rowNo, colNo);

      for (int j = 0; j < rowNo; ++j) {
        auto rowOut = row(NextState, j);
        auto rowState = row(State, j);

        auto rowRuh = row(RUH_, j);
        auto rowT   = row(Temp_, j);

        auto rowH   = subvector(rowRuh, 2 * colNo, colNo);
        auto rowT2  = subvector(rowT, 2 * colNo, colNo);

        for (int i = 0; i < colNo; ++i) {
          float ev1 = expapprox(-(rowRuh[i] + rowT[i]));
          float r = 1.0f / (1.0f + ev1);

          int k = i + colNo;
          float ev2 = expapprox(-(rowRuh[k] + rowT[k]));
          float u = 1.0f / (1.0f + ev2);

          float hv = rowH[i];
          float t2v = rowT2[i] + w_.Bx2_(0, i);
          hv = tanhapprox(hv + r * t2v);
          rowOut[i] = (1.0f - u) * hv + u * rowState[i];
        }
      }
    }
    size_t GetStateLength() const {
      return w_.U_.rows();
    }


  private:
    // Model matrices
    const Weights& w_;
    mutable mblas::Matrix WWx_;
    mutable mblas::Matrix UUx_;
    mutable mblas::Matrix Wbbx_;
    mutable mblas::Matrix lns_WWx_;
    mutable mblas::Matrix lns_UUx_;
    mutable mblas::Matrix lnb_WWx_;
    mutable mblas::Matrix lnb_UUx_;

    // reused to avoid allocation
    mutable mblas::Matrix RUH_;
    mutable mblas::Matrix RUH_1_;
    mutable mblas::Matrix RUH_2_;
    mutable mblas::Matrix Temp_;
    mutable mblas::Matrix Temp_1_;
    mutable mblas::Matrix Temp_2_;

    bool layerNormalization_;
};

}
}

