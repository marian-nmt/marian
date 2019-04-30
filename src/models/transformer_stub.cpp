#include "models/transformer.h"

namespace marian {
// factory functions
Ptr<EncoderBase> NewEncoderTransformer(Ptr<Options> options)
{
  return New<EncoderTransformer>(options);
}

Ptr<DecoderBase> NewDecoderTransformer(Ptr<Options> options)
{
  return New<DecoderTransformer>(options);
}
}  // namespace marian
