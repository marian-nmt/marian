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
      RUH_.noalias() = Context * WWx_;
      Temp_.noalias() = State * UUx_;
      
      size_t rows = State.rows();
      size_t cols = State.cols();
      
      auto R = RUH_.block(0, 0 * cols, rows, cols);
      auto U = RUH_.block(0, 1 * cols, rows, cols);
      auto H = RUH_.block(0, 2 * cols, rows, cols);
      
      auto Tr = Temp_.block(0, 0 * cols, rows, cols);
      auto Tu = Temp_.block(0, 1 * cols, rows, cols);
      auto Th = Temp_.block(0, 2 * cols, rows, cols);
      
      auto br = w_.B_.head(cols);
      auto bu = w_.B_.tail(cols);
      
      auto r = ((R + Tr).rowwise() + br).unaryExpr(&logitapprox);
      auto u = ((U + Tu).rowwise() + bu).unaryExpr(&logitapprox);
      
      auto hv = H.rowwise() + w_.Bx1_;
      auto tv = Th.rowwise() + w_.Bx2_;
      auto h = (hv.array() + (r.array() * tv.array())).unaryExpr(&tanhapprox);
      NextState = (1.0 - u.array()) * h + u.array() * State.array();
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