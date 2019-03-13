#pragma once

#include "marian.h"

#include "models/decoder.h"
#include "models/encoder.h"

namespace marian {
Ptr<EncoderBase> NewEncoderTransformer(Ptr<Options> options);
Ptr<DecoderBase> NewDecoderTransformer(Ptr<Options> options);
}  // namespace marian
