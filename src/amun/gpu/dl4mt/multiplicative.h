#pragma once
#include <boost/timer/timer.hpp>
#include "gpu/mblas/matrix_functions.h"
#include "gpu/mblas/matrix_wrapper.h"
#include "gpu/mblas/handles.h"
#include "gpu/dl4mt/cell.h"
#include "cellstate.h"
#include "gpu/dl4mt/lstm.h"
#include "gpu/dl4mt/model.h"

namespace amunmt {
namespace GPU {

template <template<class> class CellType, class InnerWeights>
class Multiplicative: public Cell {
  public:
 Multiplicative(const Weights::MultWeights<InnerWeights>& model)
      : innerCell_(model), w_(model)
    {}
    virtual void GetNextState(CellState& NextState,
                              const CellState& State,
                              const mblas::Matrix& Context) const {
      using namespace mblas;
      // TODO: the weight matrix naming probably is inconsistent
      /* HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream())); */
      /* std::cerr << "Multipliative::GetNextState1" << std::endl; */

      Copy(*tempState_.cell, *State.cell);
      Prod(*tempState_.output, *State.output, *w_.Um_);
      BroadcastVec(_1 + _2, *tempState_.output, *w_.Bmu_);
      Prod(x_mult_, Context, *w_.Wm_);
      BroadcastVec(_1 + _2, x_mult_, *w_.Bm_);
      Element(_1 * _2, *tempState_.output, x_mult_);
      innerCell_.GetNextState(NextState, tempState_, Context);
    }
    virtual CellLength GetStateLength() const {
      return innerCell_.GetStateLength();
    }
  private:
    CellType<InnerWeights> innerCell_;
    const Weights::MultWeights<InnerWeights>& w_;
    mutable mblas::Matrix x_mult_;
    mutable CellState tempState_;
};
}
}
