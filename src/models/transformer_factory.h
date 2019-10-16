// @TODO: rename to transformer.h eventually. This is not a Factory as in factory.h.
#pragma once

#include "marian.h"

#include "models/decoder.h"
#include "models/encoder.h"

namespace marian {
Ptr<EncoderBase> NewEncoderTransformer(Ptr<ExpressionGraph> graph, Ptr<Options> options);
Ptr<DecoderBase> NewDecoderTransformer(Ptr<ExpressionGraph> graph, Ptr<Options> options);
}  // namespace marian
