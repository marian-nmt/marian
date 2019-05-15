#pragma once

#include "marian.h"
#include "models/states.h"

namespace marian {

class EncoderBase : public EncoderDecoderLayerBase {
public:
  EncoderBase(Ptr<Options> options) :
    EncoderDecoderLayerBase("encoder", /*batchIndex=*/0, options,
        /*dropoutParamName=*/"dropout-src",
        /*embeddingFixParamName=*/"embedding-fix-src") {}

  // @TODO: turn into an interface. Also see if we can get rid of the graph parameter.
  virtual Ptr<EncoderState> build(Ptr<ExpressionGraph>, Ptr<data::CorpusBatch>) = 0;

  virtual void clear() = 0;
};

}  // namespace marian
