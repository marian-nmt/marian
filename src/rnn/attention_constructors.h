#pragma once

#include "marian.h"

#include "layers/factory.h"
#include "rnn/attention.h"
#include "rnn/constructors.h"
#include "rnn/types.h"

namespace marian {
namespace rnn {

class AttentionFactory : public InputFactory {
protected:
  Ptr<EncoderState> state_;

public:
  AttentionFactory(Ptr<ExpressionGraph> graph) : InputFactory(graph) {}

  Ptr<CellInput> construct() override {
    ABORT_IF(!state_, "EncoderState not set");
    return New<Attention>(graph_, options_, state_);
  }

  Accumulator<AttentionFactory> set_state(Ptr<EncoderState> state) {
    state_ = state;
    return Accumulator<AttentionFactory>(*this);
  }

  int dimAttended() {
    ABORT_IF(!state_, "EncoderState not set");
    return state_->getAttended()->shape()[1];
  }
};

typedef Accumulator<AttentionFactory> attention;
}  // namespace rnn
}  // namespace marian
