#pragma once

#include "marian.h"
#include "models/states.h"

namespace marian {

class EncoderBase : public EncoderDecoderLayerBase {
protected:
public:
  EncoderBase(Ptr<Options> options) :
    EncoderDecoderLayerBase("encoder", /*batchIndex=*/0, options) {}

  // @TODO: turn into an interface. Also see if we can get rid of the graph parameter.
  virtual Ptr<EncoderState> build(Ptr<ExpressionGraph>, Ptr<data::CorpusBatch>) = 0;

  virtual void clear() = 0;
};

}  // namespace marian
