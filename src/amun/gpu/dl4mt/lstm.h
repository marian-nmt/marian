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

      HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
      std::cerr << "SlowLSTM::GetNextState1" << std::endl;

      const size_t cols = GetStateLength().output;

      // @TODO: Optimization
      // @TODO: Launch streams to perform GEMMs in parallel
      // @TODO: Join matrices and perform single GEMM --------
      Prod(FIO_, Context, *w_.W_);

      HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
      std::cerr << "SlowLSTM::GetNextState2" << std::endl;

      Prod(H_,  Context, *w_.Wx_);
      // -----------------------------------------------------

      HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
      std::cerr << "SlowLSTM::GetNextState3" << std::endl;

      // @TODO: Join matrices and perform single GEMM --------
      Prod(Temp1_, *(State.output), *w_.U_);

      HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
      std::cerr << "SlowLSTM::GetNextState4" << std::endl;

      Prod(Temp2_, *(State.cell), *w_.Ux_);
      //std::cerr << "Temp2_=" << Temp2_.Debug(1) << std::endl;
      // -----------------------------------------------------

      HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
      std::cerr << "SlowLSTM::GetNextState5" << std::endl;

      // @TODO: Organize into one kernel ---------------------
      //std::cerr << "1RU_=" << RU_.Debug(1) << std::endl;
      //std::cerr << "w_.B_=" << w_.B_.Debug(1) << std::endl;
      BroadcastVec(_1 + _2, FIO_, *w_.B_); // Broadcasting row-wise
      //std::cerr << "2RU_=" << RU_.Debug(1) << std::endl;

      HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
      std::cerr << "SlowLSTM::GetNextState6" << std::endl;

      //std::cerr << "Temp1_=" << Temp1_.Debug(1) << std::endl;
      Element(Logit(_1 + _2), FIO_, Temp1_);
      //std::cerr << "3RU_=" << RU_.Debug(1) << std::endl;

      HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
      std::cerr << "SlowLSTM::GetNextState7" << std::endl;

      //std::cerr << "cols=" << cols << std::endl;
      Slice(F_, FIO_, 0, cols);
      //std::cerr << "R_=" << R_.Debug(1) << std::endl;

      HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
      std::cerr << "SlowLSTM::GetNextState8" << std::endl;

      Slice(I_, FIO_, 1, cols);
      //std::cerr << "U_=" << U_.Debug(1) << std::endl;

      HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
      std::cerr << "SlowLSTM::GetNextState9" << std::endl;

      Slice(O_, FIO_, 2, cols);

      HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
      std::cerr << "SlowLSTM::GetNextState10" << std::endl;

      BroadcastVec(_1 + _2, H_,    *w_.Bx1_); // Broadcasting row-wise
      //std::cerr << "H_=" << H_.Debug(1) << std::endl;

      HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
      std::cerr << "SlowLSTM::GetNextState11" << std::endl;

      //std::cerr << "1Temp2_=" << Temp2_.Debug(1) << std::endl;
      //std::cerr << "w_.Bx2_=" << w_.Bx2_.Debug(1) << std::endl;
      BroadcastVec(_1 + _2, Temp2_, *w_.Bx2_); // Broadcasting row-wise
      //std::cerr << "2Temp2_=" << Temp2_.Debug(1) << std::endl;

      HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
      std::cerr << "SlowLSTM::GetNextState12" << std::endl;

      //std::cerr << "1H_=" << H_.Debug(1) << std::endl;
      //std::cerr << "R_=" << R_.Debug(1) << std::endl;
      std::cerr << "Temp2_=" << Temp2_.Debug(0) << std::endl;
      std::cerr << "F_=" << F_.Debug(0) << std::endl;
      //Element(_1 * _3 + Tanh(_2) * _4, Temp2_, H_, F_, I_);
      Element(_1 * _2, Temp2_, F_);

      HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
      std::cerr << "SlowLSTM::GetNextState13" << std::endl;

      Element(Tanh(_1) * _2, H_, I_);

      HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
      std::cerr << "SlowLSTM::GetNextState14" << std::endl;

      Element(_1 + _2, Temp2_, H_);
      //std::cerr << "2H_=" << H_.Debug(1) << std::endl;

      HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
      std::cerr << "SlowLSTM::GetNextState15" << std::endl;

      //std::cerr << "1U_=" << U_.Debug(1) << std::endl;
      //std::cerr << "H_=" << H_.Debug(1) << std::endl;
      //std::cerr << "State=" << State.Debug(1) << std::endl;
      Element(_1 * Tanh(_2), O_, Temp2_);
      //std::cerr << "2U_=" << H_.Debug(1) << std::endl;

      HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
      std::cerr << "SlowLSTM::GetNextState16" << std::endl;
// -----------------------------------------------------

      Swap(*(NextState.output), O_);

      HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
      std::cerr << "SlowLSTM::GetNextState17" << std::endl;

      Swap(*(NextState.cell), Temp2_);

      HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream()));
      std::cerr << "SlowLSTM::GetNextState18" << std::endl;

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


