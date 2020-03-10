#include "models/transformer.h"

namespace marian {
// factory functions
Ptr<EncoderBase> NewEncoderTransformer(Ptr<ExpressionGraph> graph, Ptr<Options> options)
{
  return New<EncoderTransformer>(graph, options);
}

Ptr<DecoderBase> NewDecoderTransformer(Ptr<ExpressionGraph> graph, Ptr<Options> options)
{
  return New<DecoderTransformer>(graph, options);
}
}  // namespace marian
