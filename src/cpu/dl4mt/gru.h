#pragma once
#include "../mblas/matrix.h"

namespace amunmt {
namespace CPU {

template <class Weights>
class GRU {
  public:
    GRU(const Weights& model)
    : w_(model) {
      using namespace mblas;
      WWx_ = Concat<byColumn, Matrix>(w_.W_, w_.Wx_);
      UUx_ = Concat<byColumn, Matrix>(w_.U_, w_.Ux_);
    }
          
    void GetNextState(mblas::Matrix& NextState,
                      const mblas::Matrix& State,
                      const mblas::Matrix& Context) const {
      RUH_ = Context * WWx_;
      Temp_ = State * UUx_;
      
      // @TODO: once broadcasting is available
      // implement this using blaze idioms
      ElementwiseOps(NextState, State);
    }
          
    void ElementwiseOps(mblas::Matrix& NextState,
                        const mblas::Matrix& State) const {
      
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
    mutable mblas::Matrix WWx_;
    mutable mblas::Matrix UUx_;
    
    // reused to avoid allocation
    mutable mblas::Matrix RUH_;
    mutable mblas::Matrix Temp_;
};

}
}

