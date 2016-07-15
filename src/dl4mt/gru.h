#pragma once
#include "mblas/matrix.h"

template <class Weights>
class GRU {
  public:
    GRU(const Weights& model)
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
