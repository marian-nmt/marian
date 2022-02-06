#include "costs.h"

namespace marian {
namespace models {

Ptr<DecoderState> LogSoftmaxStep::apply(Ptr<DecoderState> state) {
  // decoder needs normalized probabilities (note: skipped if beam 1 and --skip-cost)
  state->setLogProbs(state->getLogProbs().applyUnaryFunction(logsoftmax));
  // @TODO: This is becoming more and more opaque ^^. Can we simplify this?
  return state;
}

Ptr<DecoderState> GumbelSoftmaxStep::apply(Ptr<DecoderState> state) {
  state->setLogProbs(state->getLogProbs().applyUnaryFunctions(
      [](Expr logits) {  // lemma gets gumbelled
        return logsoftmax(logits + constant_like(logits, inits::gumbel()));
      },
      logsoftmax));  // factors don't
  return state;
}

TopkGumbelSoftmaxStep::TopkGumbelSoftmaxStep(int k) : k_{k} {}

Ptr<DecoderState> TopkGumbelSoftmaxStep::apply(Ptr<DecoderState> state) {
  state->setLogProbs(state->getLogProbs().applyUnaryFunctions(
      [=](Expr logits) {  // lemma gets gumbelled
        // create logits-sized tensor consisting only of invalid path scores
        float invalidPathScore = NumericLimits<float>(logits->value_type()).lowest;
        Expr invalidLogits = constant_like(logits, inits::fromValue(invalidPathScore));
        
        // select top-k values
        Expr val, idx;
        std::tie(val, idx) = topk(logits, k_, /*axis=*/-1, /*descending=*/true);
        
        // uncomment below to display probability mass in top-k selection
        // debug(sum(gather(softmax(logits), -1, idx), -1), "sum");

        // Add Gumbel noise to top-k values only and compute logsoftmax, used for argmax sampling later in beam-search
        Expr gumbelVal = logsoftmax(val + constant_like(val, inits::gumbel()));

        // Scatter gumbelled values back into logits to fill with usable values
        return scatter(invalidLogits, -1, idx, gumbelVal);
      },
      logsoftmax));  // factors don't
  return state;
}

}  // namespace models
}  // namespace marian
