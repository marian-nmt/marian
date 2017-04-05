#pragma once

#include "gpu/mblas/matrix_functions.h"

namespace amunmt {
namespace GPU {

template <class Weights>
class SlowGRU {
  public:
    SlowGRU(const Weights& model)
    : w_(model) {}

    void GetNextState(mblas::Matrix& NextState,
                      const mblas::Matrix& State,
                      const mblas::Matrix& Context) const {
      using namespace mblas;

      std::cerr << std::endl;

      const size_t cols = GetStateLength();

      // @TODO: Optimization
      // @TODO: Launch streams to perform GEMMs in parallel
      // @TODO: Join matrices and perform single GEMM --------
      Prod(RU_, Context, w_.W_);
      Prod(H_,  Context, w_.Wx_);
      // -----------------------------------------------------

      // @TODO: Join matrices and perform single GEMM --------
      Prod(Temp1_, State, w_.U_);
      Prod(Temp2_, State, w_.Ux_);
      //std::cerr << "Temp2_=" << Temp2_.Debug(1) << std::endl;
      // -----------------------------------------------------

      // @TODO: Organize into one kernel ---------------------
      std::cerr << "1RU_=" << RU_.Debug(1) << std::endl;
      std::cerr << "w_.B_=" << w_.B_.Debug(1) << std::endl;
      BroadcastVec(_1 + _2, RU_, w_.B_); // Broadcasting row-wise
      std::cerr << "2RU_=" << RU_.Debug(1) << std::endl;

      std::cerr << "Temp1_=" << Temp1_.Debug(1) << std::endl;
      Element(Logit(_1 + _2), RU_, Temp1_);
      std::cerr << "3RU_=" << RU_.Debug(1) << std::endl;

      std::cerr << "cols=" << cols << std::endl;
      Slice(R_, RU_, 0, cols);
      std::cerr << "R_=" << R_.Debug(1) << std::endl;

      Slice(U_, RU_, 1, cols);
      std::cerr << "U_=" << U_.Debug(1) << std::endl;

      //abort();

      BroadcastVec(_1 + _2, H_,    w_.Bx1_); // Broadcasting row-wise
      std::cerr << "H_=" << H_.Debug(1) << std::endl;

      std::cerr << "1Temp2_=" << Temp2_.Debug(1) << std::endl;
      std::cerr << "w_.Bx2_=" << w_.Bx2_.Debug(1) << std::endl;
      BroadcastVec(_1 + _2, Temp2_, w_.Bx2_); // Broadcasting row-wise
      std::cerr << "2Temp2_=" << Temp2_.Debug(1) << std::endl;

      Element(Tanh(_1 + _2 * _3), H_, R_, Temp2_);
      Element((1.0 - _1) * _2 + _1 * _3, U_, H_, State);
      // -----------------------------------------------------

      Swap(NextState, U_);
    }

    size_t GetStateLength() const {
      return w_.U_.dim(0);
    }

  private:
    // Model matrices
    const Weights& w_;

    // reused to avoid allocation
    mutable mblas::Matrix RU_;
    mutable mblas::Matrix R_;
    mutable mblas::Matrix U_;
    mutable mblas::Matrix H_;
    mutable mblas::Matrix Temp1_;
    mutable mblas::Matrix Temp2_;

    SlowGRU(const SlowGRU&) = delete;
};

///////////////////////////////////////////////////////////////////////////////////////////////

__global__ void gElementwiseOps(float* out,
                                const float* state,
                                const float* ruh,
                                const float* t,
                                const float* b,
                                const float* bx1,
                                const float* bx2,
                                size_t rows, size_t cols);

template <class Weights>
class FastGRU {
  public:
    FastGRU(const Weights& model)
    : w_(model) {
      /*for(int i = 0; i < 4; ++i) {
        cudaStreamCreate(&s_[i]);
        cublasCreate(&h_[i]);
        cublasSetStream(h_[i], s_[i]);
      }*/

      using namespace mblas;
      Transpose(WWx_, w_.W_);
      std::cerr << std::endl;
      std::cerr << "w_.W_=" << w_.W_.Debug(1) << std::endl;
      std::cerr << "1WWx_=" << WWx_.Debug(1) << std::endl;

      Matrix WxT;
      Transpose(WxT, w_.Wx_);
      std::cerr << "w_.Wx_=" << w_.Wx_.Debug(1) << std::endl;
      std::cerr << "WxT=" << WxT.Debug(1) << std::endl;

      Concat(WWx_, WxT);
      std::cerr << "2WWx_=" << WWx_.Debug(1) << std::endl;

      Transpose(WWx_);
      std::cerr << "3WWx_=" << WWx_.Debug(1) << std::endl;

      Transpose(UUx_, w_.U_);
      Matrix UxT;
      Transpose(UxT, w_.Ux_);
      Concat(UUx_, UxT);
      Transpose(UUx_);

      std::cerr << std::endl;
    }

    void GetNextState(mblas::Matrix& NextState,
                      const mblas::Matrix& State,
                      const mblas::Matrix& Context) const {
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

      Prod(Temp_, State, UUx_);
      std::cerr << "State=" << State.Debug(1) << std::endl;
      std::cerr << "UUx_" << UUx_.Debug(1) << std::endl;
      std::cerr << "Temp_=" << Temp_.Debug(1) << std::endl;

      if (w_.Gamma_2_) {
        Normalization(Temp_, Temp_, w_.Gamma_2_, 1e-9);
      }

      ElementwiseOps(NextState, State, RUH_, Temp_);
    }


    void ElementwiseOps(mblas::Matrix& NextState,
                        const mblas::Matrix& State,
                        const mblas::Matrix& RUH,
                        const mblas::Matrix& Temp) const {
      const size_t rows = State.dim(0) * State.dim(2) * State.dim(3);
      const size_t cols = State.dim(1);

      NextState.Resize(State.dim(0) * State.dim(3), cols, State.dim(2), 1);
      //std::cerr << "NextState=" << NextState.Debug() << std::endl;

      int blocks  = std::min(MAX_BLOCKS, (int)rows);
      int threads = std::min(MAX_THREADS, (int)cols);
      gElementwiseOps<<<blocks, threads, 0, mblas::CudaStreamHandler::GetStream()>>>
        (NextState.data(), State.data(), RUH.data(), Temp.data(), w_.B_.data(), w_.Bx1_.data(),
         w_.Bx2_.data(), rows, cols);
    }

    size_t GetStateLength() const {
      return w_.U_.dim(0);
    }


  private:
    // Model matrices
    const Weights& w_;

    // reused to avoid allocation
    mutable mblas::Matrix WWx_;
    mutable mblas::Matrix UUx_;

    mutable mblas::Matrix RUH_;
    mutable mblas::Matrix Temp_;

    FastGRU(const FastGRU&) = delete;
};

template<class T>
using GRU = SlowGRU<T>;

}
}


