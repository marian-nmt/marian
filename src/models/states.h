#pragma once

#include "common/definitions.h"
#include "data/corpus.h"
#include "graph/expression_graph.h"

namespace marian {

struct EncoderState {
  virtual Expr getContext() = 0;
  virtual Expr getAttended() = 0;
  virtual Expr getMask() = 0;
  virtual const std::vector<size_t>& getSourceWords() = 0;
};

class DecoderState {
private:
  Expr targetEmbeddings_;
  bool singleStep_{false};

public:
  virtual Ptr<EncoderState> getEncoderState() = 0;

  virtual Expr getProbs() = 0;
  virtual void setProbs(Expr) = 0;

  virtual Expr getTargetEmbeddings() { return targetEmbeddings_; };

  virtual void setTargetEmbeddings(Expr targetEmbeddings) {
    targetEmbeddings_ = targetEmbeddings;
  }

  virtual bool doSingleStep() { return singleStep_; };

  virtual void setSingleStep(bool singleStep = true) {
    singleStep_ = singleStep;
  }

  virtual Ptr<DecoderState> select(const std::vector<size_t>&) = 0;

  virtual const std::vector<size_t>& getSourceWords() {
    return getEncoderState()->getSourceWords();
  }

  virtual void blacklist(Expr totalCosts, Ptr<data::CorpusBatch> batch) {}
};
}