#pragma once

#include "cpu/mblas/matrix.h"
#include "model.h"

namespace amunmt {
namespace CPU {
namespace Nematus {

class Transition {
  public:
    Transition(const Weights::Transition& model);

    void GetNextState(mblas::Matrix& state) const;

  protected:
    void ElementwiseOps(mblas::Matrix& state, int idx) const;

  private:
    // Model matrices
    const Weights::Transition& w_;

    // reused to avoid allocation
    mutable mblas::Matrix UUx_;
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
}

