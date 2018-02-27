#pragma once

#include "cpu/mblas/tensor.h"
#include "model.h"

namespace amunmt {
namespace CPU {
namespace Nematus {

class Transition {
  public:
    Transition(const Weights::Transition& model);

    void GetNextState(mblas::Tensor& state) const;

  protected:
    void ElementwiseOps(mblas::Tensor& state, int idx) const;

  private:
    // Model matrices
    const Weights::Transition& w_;

    // reused to avoid allocation
    mutable mblas::Tensor UUx_;
    mutable mblas::Tensor RUH_;
    mutable mblas::Tensor RUH_1_;
    mutable mblas::Tensor RUH_2_;
    mutable mblas::Tensor Temp_;
    mutable mblas::Tensor Temp_1_;
    mutable mblas::Tensor Temp_2_;

    bool layerNormalization_;
};

}
}
}

