
#include "models/lex_probs.h"

namespace marian {

thread_local Ptr<sparse::CSR> LexProbs::lexProbs_;
thread_local Ptr<sparse::CSR> LexProbs::lf_;

Expr LexicalBias::operator()(Expr logits) {
  auto& alignmentsVec = attention_->getAlignments();

  Expr aln;
  if(single_)
    aln = alignmentsVec.back();
  else
    aln = concatenate(alignmentsVec, keywords::axis = 3);

  return lexical_bias(logits, aln, eps_, sentLexProbs_);
}
}
