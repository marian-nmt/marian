#pragma once
#include "mblas/matrix.h"

template <class Weights>
class SlowGRU {
  public:
    SlowGRU(const Weights& model)
    : w_(model) {}
          
    void GetNextState(mblas::Matrix& NextState,
                      const mblas::Matrix& State,
                      const mblas::Matrix& Context) const {
      using namespace mblas;
      namespace bpp = boost::phoenix::placeholders;
      
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
      // -----------------------------------------------------
      
      // @TODO: Organize into one kernel ---------------------
      BroadcastVec(bpp::_1 + bpp::_2, RU_, w_.B_); // Broadcasting row-wise
      Element(Logit(bpp::_1 + bpp::_2), RU_, Temp1_);
      Slice(R_, RU_, 0, cols);
      Slice(U_, RU_, 1, cols);
      
      BroadcastVec(bpp::_1 + bpp::_2, H_,    w_.Bx1_); // Broadcasting row-wise
      BroadcastVec(bpp::_1 + bpp::_2, Temp2_, w_.Bx2_); // Broadcasting row-wise
      
      Element(Tanh(bpp::_1 + bpp::_2 * bpp::_3), H_, R_, Temp2_);
      Element((1.0 - bpp::_1) * bpp::_2 + bpp::_1 * bpp::_3, U_, H_, State);
      // -----------------------------------------------------
      
      NextState.swap(U_);
    }
    
    size_t GetStateLength() const {
      return w_.U_.rows();
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
};

void gElementwiseOps(float* out,
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
      WWx_.resize(w_.W_.rows(),
                  w_.W_.cols() + w_.Wx_.cols());
      WWx_ << w_.W_, w_.Wx_;
      
      UUx_.resize(w_.U_.rows(),
                  w_.U_.cols() + w_.Ux_.cols());
      UUx_ << w_.U_, w_.Ux_; 
    }
          
    void GetNextState(mblas::Matrix& NextState,
                      const mblas::Matrix& State,
                      const mblas::Matrix& Context) const {
      using namespace mblas;
      RUH_ = Context * WWx_;
      Temp_ = State * UUx_;
      ElementwiseOps(NextState, State, RUH_, Temp_);
    }
          
    void ElementwiseOps(mblas::Matrix& NextState,
                        const mblas::Matrix& State,
                        const mblas::Matrix& RUH,
                        const mblas::Matrix& Temp) const {
      
      const size_t rows = State.rows();
      const size_t cols = State.cols();
      NextState.resize(rows, cols);
      
      float* out = NextState.data();
      const float* state = State.data();
      const float* ruh = RUH.data();
      const float* t = Temp.data();
      const float* br = w_.B_.data();
      const float* bu = br + cols;
      const float* bx1 = w_.Bx1_.data();
      const float* bx2 = w_.Bx2_.data();
      
      size_t shift = cols * rows;
      
      #pragma omp for schedule(dynamic, 100)
      for(int j = 0; j < cols; ++j) {
        float* colOut = out + j * rows;
        const float* colR = ruh + j * rows;
        const float* colU = colR + shift;
        const float* colH = colU + shift;
        
        const float* colTr = t + j * rows;
        const float* colTu = colTr + shift;
        const float* colTh = colTu + shift;
        
        const float* colState = state + j * rows;
        
        for(int i = 0; i < rows; ++i) {
          float ev1 = expapprox(-(colR[i] + br[j] + colTr[i]));
          float r = 1.0 / (1.0 + ev1);
          
          float ev2 = expapprox(-(colU[i] + bu[j] + colTu[i]));
          float u = 1.0 / (1.0 + ev2);              
    
          float hv = colH[i] + bx1[j];
          float t2v = colTh[i] + bx2[j];
          hv = tanhapprox(hv + r * t2v);
          colOut[i] = (1.0 - u) * hv + u * colState[i];
        }
      }
    }
    
    size_t GetStateLength() const {
      return w_.U_.rows();
    }

    
  private:
    // Model matrices
    const Weights& w_;
        
    // reused to avoid allocation    
    mutable mblas::Matrix WWx_;
    mutable mblas::Matrix UUx_;
    
    mutable mblas::Matrix RUH_;
    mutable mblas::Matrix Temp_;
};

template<class T>
using GRU = FastGRU<T>;