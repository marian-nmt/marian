#include "transition.h"

namespace amunmt {
namespace CPU {
namespace Nematus {

Transition::Transition(const Weights::Transition& model)
  : w_(model),
    layerNormalization_(false)
{
  if (w_.U_lns_.size() > 1 && w_.U_lns_[0].rows() > 1) {
    layerNormalization_ = true;
  }
}


void Transition::GetNextState(mblas::Matrix& state) const
{
  if (layerNormalization_) {
    for (int i = 0; i < w_.size(); ++i) {
      Temp_1_ = state * w_.U_[i];
      Temp_2_ = state * w_.Ux_[i];

      switch(w_.type()) {
        case Weights::Transition::TransitionType::Encoder:
          LayerNormalization(Temp_1_, w_.U_lns_[i], w_.U_lnb_[i]);
          mblas::AddBiasVector<mblas::byRow>(Temp_1_, w_.B_[i]);

          LayerNormalization(Temp_2_, w_.Ux_lns_[i], w_.Ux_lnb_[i]);
          break;

        case Weights::Transition::TransitionType::Decoder:
          mblas::AddBiasVector<mblas::byRow>(Temp_1_, w_.B_[i]);
          LayerNormalization(Temp_1_, w_.U_lns_[i], w_.U_lnb_[i]);

          mblas::AddBiasVector<mblas::byRow>(Temp_2_, w_.Bx1_[i]);
          LayerNormalization(Temp_2_, w_.Ux_lns_[i], w_.Ux_lnb_[i]);
          break;
      }
      ElementwiseOps(state, i);
    }
  } else {
    for (int i = 0; i < w_.size(); ++i) {
      Temp_1_ = state * w_.U_[i];
      Temp_2_ = state * w_.Ux_[i];
      mblas::AddBiasVector<mblas::byRow>(Temp_1_, w_.B_[i]);
      mblas::AddBiasVector<mblas::byRow>(Temp_2_, w_.Bx1_[i]);
      ElementwiseOps(state, i);
    }
  }
}


void Transition::ElementwiseOps(mblas::Matrix& state, int idx) const {
  using namespace mblas;
  using namespace blaze;

  for (int j = 0; j < (int)state.dim(0); ++j) {
    auto rowState = row(state, j);
    auto rowT   = row(Temp_1_, j);
    auto rowT2  = row(Temp_2_, j);

    for (int i = 0; i < (int)state.dim(1); ++i) {
      float ev1 = expapprox(-(rowT[i])); // + w_.B_[idx](0, i)));
      float r = 1.0f / (1.0f + ev1);

      int k = i + state.dim(1);
      float ev2 = expapprox(-(rowT[k])); // + w_.B_[idx](0, k)));
      float u = 1.0f / (1.0f + ev2);

      float hv = w_.Bx2_[idx](0, i);
      float t2v = rowT2[i]; // + w_.Bx1_[idx](0, i);
      hv = tanhapprox(hv + r * t2v);
      rowState[i] = (1.0f - u) * hv + u * rowState[i];
    }
  }
}

}  // namespace Nematus
}  // namespace CPU
}  // namespace amunmt

