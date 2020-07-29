#pragma once

#include "marian.h"
#include "models/states.h"

namespace marian {

class EncoderBase : public EncoderDecoderLayerBase {
public:
  EncoderBase(Ptr<ExpressionGraph> graph, Ptr<Options> options) :
    EncoderDecoderLayerBase(graph, options, "encoder", /*batchIndex=*/0,
        options->get<float>("dropout-src", 0.0f),
        options->get<bool>("embedding-fix-src", false)) {}

  // @TODO: turn into an interface. Also see if we can get rid of the graph parameter.
  virtual Ptr<EncoderState> build(Ptr<ExpressionGraph>, Ptr<data::CorpusBatch>) = 0;

  virtual void clear() = 0;
};

}  // namespace marian
