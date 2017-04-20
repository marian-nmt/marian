#pragma once

#include "common/definitions.h"
#include "graph/expression_graph.h"

namespace marian {

struct EncoderState {
  virtual Expr getContext() = 0;
  virtual Expr getMask() = 0;
};

struct DecoderState {
  virtual Ptr<EncoderState> getEncoderState() = 0;
  virtual Expr getProbs() = 0;
  virtual void setProbs(Expr) = 0;
  virtual Ptr<DecoderState> select(const std::vector<size_t>&) = 0;
};

}