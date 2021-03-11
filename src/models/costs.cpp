#include "costs.h"

namespace marian {
namespace models {

Ptr<DecoderState> LogSoftmaxStep::apply(Ptr<DecoderState> state) {
  // decoder needs normalized probabilities (note: skipped if beam 1 and --skip-cost)
  state->setLogProbs(state->getLogProbs().applyUnaryFunction(logsoftmax));
  // @TODO: This is becoming more and more opaque ^^. Can we simplify this?
  return state;
}

}  // namespace models
}  // namespace marian
