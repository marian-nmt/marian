#pragma once
#include <boost/timer/timer.hpp>
#include "gpu/mblas/tensor_functions.h"
#include "gpu/mblas/tensor_wrapper.h"
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
                      const mblas::Tensor& Context) const {
      using namespace mblas;

      /* HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream())); */
      /* std::cerr << "SlowLSTM::GetNextState1" << std::endl; */

      const unsigned cols = GetStateLength().output;

      // transform context for use with gates and for computing the input (the C part)
      Prod(FIOC_, Context, *w_.W_);
      BroadcastVec(_1 + _2, FIOC_, *w_.B_); // Broadcasting row-wise

      // transform previous output for use with gates and for computing the input
      Prod(Temp1_, *(State.output), *w_.U_);

      Element(_1 + _2, FIOC_, Temp1_);
      // compute the gates
      Slice(FIO_, FIOC_, 0, cols * 3);
      Element(Logit(_1), FIO_);
      Slice(F_, FIO_, 0, cols);
      Slice(I_, FIO_, 1, cols);
      Slice(O_, FIO_, 2, cols);
      // compute the input
      Slice(C_, FIOC_, 3, cols);
      Element(Tanh(_1), C_);

      // apply the forget gate
      Copy(*NextState.cell, *State.cell);
      Element(_1 * _2, *NextState.cell, F_);
      // apply the input gate
      Element(_1 * _2, C_, I_);
      // update the cell state with the input
      Element(_1 + _2, *NextState.cell, C_);
      // apply the output gate
      Element(_1 * Tanh(_2), O_, *NextState.cell);
      Swap(*(NextState.output), O_);
    }

    virtual CellLength GetStateLength() const {
      return CellLength(w_.U_->dim(0), w_.U_->dim(0));
    }

    virtual std::string Debug(unsigned verbosity = 1) const
    {
      return "LSTM";
    }

  private:
    // Model matrices
    const Weights& w_;

    // reused to avoid allocation
    mutable mblas::Tensor FIO_;
    mutable mblas::Tensor FIOC_;
    mutable mblas::Tensor F_;
    mutable mblas::Tensor I_;
    mutable mblas::Tensor O_;
    mutable mblas::Tensor C_;
    mutable mblas::Tensor Temp1_;
    mutable mblas::Tensor Temp2_;

    SlowLSTM(const SlowLSTM&) = delete;
};

template<class T>
using LSTM = SlowLSTM<T>;

}
}


