#pragma once
#include <boost/timer/timer.hpp>
#include "gpu/mblas/matrix_functions.h"
#include "gpu/mblas/matrix_wrapper.h"
#include "gpu/mblas/handles.h"
#include "gpu/dl4mt/cell.h"
#include "cellstate.h"

namespace amunmt {
namespace GPU {

template <class Weights>
class SlowLSTM: public Cell {
  public:
    SlowLSTM(const Weights& model)
    : w_(model) {}

    virtual void GetNextState(CellState& NextState,
                      const CellState& State,
                      const mblas::Matrix& Context) const {
      using namespace mblas;

      /* HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream())); */
      /* std::cerr << "SlowLSTM::GetNextState1" << std::endl; */

      const size_t cols = GetStateLength().output;

      // transform context for use with gates
      Prod(FIO_, Context, *w_.W_);
      BroadcastVec(_1 + _2, FIO_, *w_.B_); // Broadcasting row-wise
      // transform context for use with computing the input
      Prod(H_,  Context, *w_.Wx_);
      BroadcastVec(_1 + _2, H_, *w_.Bx_); // Broadcasting row-wise

      // transform previous output for use with gates
      Prod(Temp1_, *(State.output), *w_.U_);
      // transform previous output for use with computing the input
      Prod(Temp2_, *(State.output), *w_.Ux_);

      // compute the gates
      Element(Logit(_1 + _2), FIO_, Temp1_);
      Slice(F_, FIO_, 0, cols);
      Slice(I_, FIO_, 1, cols);
      Slice(O_, FIO_, 2, cols);
      // compute the input
      Element(Tanh(_1 + _2), H_, Temp2_);

      // apply the forget gate
      Copy(*NextState.cell, *State.cell);
      Element(_1 * _2, *NextState.cell, F_);
      // apply the input gate
      Element(_1 * _2, H_, I_);
      // update the cell state with the input
      Element(_1 + _2, *NextState.cell, H_);
      // apply the output gate
      Element(_1 * Tanh(_2), O_, *NextState.cell);
      Swap(*(NextState.output), O_);
    }

    virtual CellLength GetStateLength() const {
      return CellLength(w_.U_->dim(0), w_.U_->dim(0));
    }

  private:
    // Model matrices
    const Weights& w_;

    // reused to avoid allocation
    mutable mblas::Matrix FIO_;
    mutable mblas::Matrix F_;
    mutable mblas::Matrix I_;
    mutable mblas::Matrix O_;
    mutable mblas::Matrix H_;
    mutable mblas::Matrix Temp1_;
    mutable mblas::Matrix Temp2_;

    SlowLSTM(const SlowLSTM&) = delete;
};

template<class T>
using LSTM = SlowLSTM<T>;

}
}


