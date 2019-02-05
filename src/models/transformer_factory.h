#pragma once

#include "marian.h"

#include "models/decoder.h"
#include "models/encoder.h"
//#include "models/states.h"
//#include "layers/constructors.h"
//#include "layers/factory.h"

namespace marian {
// @TODO: find out why static is required here to get to compile
static Ptr<EncoderBase> NewEncoderTransformer(Ptr<Options> options);
static Ptr<DecoderBase> NewDecoderTransformer(Ptr<Options> options);
}  // namespace marian
