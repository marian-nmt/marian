#include "output.h"
#include "common/timer.h"
#include "data/factored_vocab.h"
#include "layers/loss.h"
#include "layers/lsh.h"

namespace marian {
namespace mlp {

/*private*/ void Output::lazyConstruct(int inputDim) {
  // We must construct lazily since we won't know tying nor input dim in constructor.
  if(Wt_)
    return;

  // this option is only set in the decoder
  if(!lsh_ && options_->hasAndNotEmpty("output-approx-knn")) {
    auto k = opt<std::vector<int>>("output-approx-knn")[0];
    auto nbits = opt<std::vector<int>>("output-approx-knn")[1];
    lsh_ = New<LSH>(k, nbits);
  }

  auto name = options_->get<std::string>("prefix");
  auto numOutputClasses = options_->get<int>("dim");

  factoredVocab_ = FactoredVocab::tryCreateAndLoad(options_->get<std::string>("vocab", ""));
  if(factoredVocab_) {
    numOutputClasses = (int)factoredVocab_->factorVocabSize();
    LOG_ONCE(info, "[embedding] Factored outputs enabled");
  }

  if(tiedParam_) {
    Wt_ = tiedParam_;
  } else {
    if(graph_->get(name + "_W")) {  // support of legacy models that did not transpose
      Wt_ = graph_->param(
          name + "_W", {inputDim, numOutputClasses}, inits::glorotUniform(true, false));
      isLegacyUntransposedW = true;
    } else  // this is the regular case:
      Wt_ = graph_->param(
          name + "_Wt", {numOutputClasses, inputDim}, inits::glorotUniform(false, true));
  }

  if(hasBias_)
    b_ = graph_->param(name + "_b", {1, numOutputClasses}, inits::zeros());

  /*const*/ int lemmaDimEmb = options_->get<int>("lemma-dim-emb", 0);
  ABORT_IF(lemmaDimEmb && !factoredVocab_, "--lemma-dim-emb requires a factored vocabulary");
  if(lemmaDimEmb > 0) {  // > 0 means to embed the (expected) word with a different embedding matrix
#define HARDMAX_HACK
#ifdef HARDMAX_HACK
    lemmaDimEmb = lemmaDimEmb & 0xfffffffe;  // hack to select hard-max: use an odd number
#endif
    auto range = factoredVocab_->getGroupRange(0);
    auto lemmaVocabDim = (int)(range.second - range.first);
    auto initFunc = inits::glorotUniform(
        /*fanIn=*/true, /*fanOut=*/false);  // -> embedding vectors have roughly unit length
    lemmaEt_ = graph_->param(name + "_lemmaEt",
                             {lemmaDimEmb, lemmaVocabDim},
                             initFunc);  // [L x U] L=lemmaDimEmb; transposed for speed
  }
}

Logits Output::applyAsLogits(Expr input) /*override final*/ {
  lazyConstruct(input->shape()[-1]);

  auto affineOrDot = [](Expr x, Expr W, Expr b, bool transA, bool transB) {
    if(b)
      return affine(x, W, b, transA, transB);
    else
      return dot(x, W, transA, transB);
  };

  auto affineOrLSH = [this, affineOrDot](Expr x, Expr W, Expr b, bool transA, bool transB) {
    if(lsh_) {
      ABORT_IF(transA, "Transposed query not supported for LSH");
      ABORT_IF(!transB, "Untransposed indexed matrix not supported for LSH");
      return lsh_->apply(x, W, b);  // knows how to deal with undefined bias
    } else {
      return affineOrDot(x, W, b, transA, transB);
    }
  };

  if(shortlist_ && !cachedShortWt_) {  // shortlisted versions of parameters are cached within one
                                       // batch, then clear()ed
    cachedShortWt_ = index_select(Wt_, isLegacyUntransposedW ? -1 : 0, shortlist_->indices());
    if(hasBias_)
      cachedShortb_ = index_select(b_, -1, shortlist_->indices());
  }

  if(factoredVocab_) {
    auto graph = input->graph();

    // project each factor separately
    auto numGroups = factoredVocab_->getNumGroups();
    std::vector<Ptr<RationalLoss>> allLogits(numGroups,
                                             nullptr);  // (note: null entries for absent factors)
    Expr input1 = input;                                // [B... x D]
    Expr Plemma = nullptr;                              // used for lemmaDimEmb=-1
    Expr inputLemma = nullptr;                          // used for lemmaDimEmb=-2, -3
    for(size_t g = 0; g < numGroups; g++) {
      auto range = factoredVocab_->getGroupRange(g);
      if(g > 0 && range.first == range.second)  // empty entry
        continue;
      ABORT_IF(g > 0 && range.first != factoredVocab_->getGroupRange(g - 1).second,
               "Factor groups must be consecutive (group {} vs predecessor)",
               g);
      // slice this group's section out of W_
      Expr factorWt, factorB;
      if(g == 0 && shortlist_) {
        factorWt = cachedShortWt_;
        factorB = cachedShortb_;
      } else {
        factorWt = slice(
            Wt_, isLegacyUntransposedW ? -1 : 0, Slice((int)range.first, (int)range.second));
        if(hasBias_)
          factorB = slice(b_, -1, Slice((int)range.first, (int)range.second));
      }
      /*const*/ int lemmaDimEmb = options_->get<int>("lemma-dim-emb", 0);
      if((lemmaDimEmb == -2 || lemmaDimEmb == -3)
         && g > 0) {  // -2/-3 means a gated transformer-like structure (-3 = hard-max)
        LOG_ONCE(info, "[embedding] using lemma conditioning with gate");
        // this mimics one transformer layer
        //  - attention over two inputs:
        //     - e = current lemma. We use the original embedding vector; specifically, expectation
        //     over all lemmas.
        //     - input = hidden state FF(h_enc+h_dec)
        //  - dot-prod attention to allow both sides to influence (unlike our recurrent
        //  self-attention)
        //  - multi-head to allow for multiple conditions to be modeled
        //  - add & norm, for gradient flow and scaling
        //  - FF layer   --this is expensive; it is per-factor
        // multi-head attention
        int inputDim = input->shape()[-1];
        int heads = 8;
        auto name = options_->get<std::string>("prefix") + "_factor" + std::to_string(g);
        auto Wq = graph_->param(name + "_Wq", {inputDim, inputDim}, inits::glorotUniform());
        auto Wk = graph_->param(name + "_Wk", {inputDim, inputDim}, inits::glorotUniform());
        auto Wv = graph_->param(name + "_Wv", {inputDim, inputDim}, inits::glorotUniform());
        auto toMultiHead = [&](Expr x, int heads) {
          const auto& shape = x->shape();
          int inputDim = shape[-1];
          int otherDim = shape.elements() / inputDim;
          ABORT_IF(inputDim / heads * heads != inputDim,
                   "inputDim ({}) must be multiple of number of heads ({})",
                   inputDim,
                   heads);
          return reshape(x, {otherDim, heads, 1, inputDim / heads});
        };
        input1 = inputLemma;
        auto qm = toMultiHead(dot(input1, Wq), heads);  // [B... x H x D/H] projected query
        auto kdm = toMultiHead(dot(input1 - input, Wk),
                               heads);  // [B... x H x D/H] the two data vectors projected as keys.
                                        // Use diff and sigmoid, instead of softmax.
        auto vem = toMultiHead(
            dot(input1, Wv),
            heads);  // [B... x H x D/H] one of the two data vectors projected as values
        auto vim = toMultiHead(dot(input, Wv), heads);  // [B... x H x D/H] the other
        auto zm = bdot(qm, kdm, false, true);           // [B... x H x 1]
        auto sm = sigmoid(zm);                          // [B... x H x 1]
        auto rm = sm * (vem - vim) + vim;               // [B... x H x D/H]
        auto r = reshape(rm, input->shape());           // [B... x D]
        // add & norm
        input1 = r + input1;
        input1 = layerNorm(input1, name + "_att");
        // FF layer
        auto ffnDropProb = 0.1f;     // @TODO: get as a parameter
        auto ffnDim = inputDim * 2;  // @TODO: get as a parameter
        auto f = denseInline(input1,
                             name + "_ffn",
                             /*suffix=*/"1",
                             ffnDim,
                             inits::glorotUniform(),
                             (ActivationFunction*)relu,
                             ffnDropProb);
        f = denseInline(f, name + "_ffn", /*suffix=*/"2", inputDim);
        // add & norm
        input1 = f + input1;
        input1 = layerNorm(input1, name + "_ffn");
      }
      // @TODO: b_ should be a vector, not a matrix; but shotlists use cols() in, which requires a
      // matrix
      Expr factorLogits;
      if(g == 0)
        factorLogits = affineOrLSH(
            input1,
            factorWt,
            factorB,
            false,
            /*transB=*/isLegacyUntransposedW ? false : true);  // [B... x U] factor logits
      else
        factorLogits = affineOrDot(
            input1,
            factorWt,
            factorB,
            false,
            /*transB=*/isLegacyUntransposedW ? false : true);  // [B... x U] factor logits

      // optionally add lemma-dependent bias
      if(Plemma) {  // [B... x U0]
        int lemmaVocabDim = Plemma->shape()[-1];
        int factorVocabDim = factorLogits->shape()[-1];
        auto name = options_->get<std::string>("prefix");
        Expr lemmaBt
            = graph_->param(name + "_lemmaBt_" + std::to_string(g),
                            {factorVocabDim, lemmaVocabDim},
                            inits::zeros());  // [U x U0] U0=#lemmas one bias per class per lemma
        auto b = dot(Plemma, lemmaBt, false, true);  // [B... x U]
        factorLogits = factorLogits + b;
      }
      allLogits[g] = New<RationalLoss>(factorLogits, nullptr);
      // optionally add a soft embedding of lemma back to create some lemma dependency
      // @TODO: if this works, move it into lazyConstruct
      if(lemmaDimEmb == -2 && g == 0) {  // -2 means a gated transformer-like structure
        LOG_ONCE(info, "[embedding] using lemma conditioning with gate, soft-max version");
        // get expected lemma embedding vector
        auto factorLogSoftmax = logsoftmax(
            factorLogits);  // [B... x U] note: with shortlist, this is not the full lemma set
        auto factorSoftmax = exp(factorLogSoftmax);
        inputLemma = dot(factorSoftmax,
                         factorWt,
                         false,
                         /*transB=*/isLegacyUntransposedW ? true : false);  // [B... x D]
      } else if(lemmaDimEmb == -3 && g == 0) {  // same as -2 except with hard max
        LOG_ONCE(info, "[embedding] using lemma conditioning with gate, hard-max version");
        // get max-lemma embedding vector
        auto maxVal = max(factorLogits,
                          -1);  // [B... x U] note: with shortlist, this is not the full lemma set
        auto factorHardmax = eq(factorLogits, maxVal);
        inputLemma = dot(factorHardmax,
                         factorWt,
                         false,
                         /*transB=*/isLegacyUntransposedW ? true : false);  // [B... x D]
      } else if(lemmaDimEmb == -1 && g == 0) {  // -1 means learn a lemma-dependent bias
        ABORT_IF(shortlist_, "Lemma-dependent bias with short list is not yet implemented");
        LOG_ONCE(info, "[embedding] using lemma-dependent bias");
        auto factorLogSoftmax
            = logsoftmax(factorLogits);  // (we do that again later, CSE will kick in)
        auto z = /*stopGradient*/ (factorLogSoftmax);
        Plemma = exp(z);                      // [B... x U]
      } else if(lemmaDimEmb > 0 && g == 0) {  // > 0 means learn a re-embedding matrix
        LOG_ONCE(info, "[embedding] enabled re-embedding of lemma, at dim {}", lemmaDimEmb);
        // compute softmax. We compute logsoftmax() separately because this way, computation will be
        // reused later via CSE
        auto factorLogSoftmax = logsoftmax(factorLogits);
        auto factorSoftmax = exp(factorLogSoftmax);
#ifdef HARDMAX_HACK
        bool hardmax = (lemmaDimEmb & 1)
                       != 0;  // odd value triggers hardmax for now (for quick experimentation)
        if(hardmax) {
          lemmaDimEmb = lemmaDimEmb & 0xfffffffe;
          LOG_ONCE(info, "[embedding] HARDMAX_HACK enabled. Actual dim is {}", lemmaDimEmb);
          auto maxVal = max(factorSoftmax, -1);
          factorSoftmax = eq(factorSoftmax, maxVal);
        }
#endif
        // re-embedding lookup, soft-indexed by softmax
        if(shortlist_ && !cachedShortLemmaEt_)  // short-listed version of re-embedding matrix
          cachedShortLemmaEt_ = index_select(lemmaEt_, -1, shortlist_->indices());
        auto e = dot(factorSoftmax,
                     cachedShortLemmaEt_ ? cachedShortLemmaEt_ : lemmaEt_,
                     false,
                     true);  // [B... x L]
        // project it back to regular hidden dim
        int inputDim = input1->shape()[-1];
        auto name = options_->get<std::string>("prefix");
        // note: if the lemmaEt[:,w] have unit length (var = 1/L), then lemmaWt @ lemmaEt is also
        // length 1
        Expr lemmaWt
            = inputDim == lemmaDimEmb
                  ? nullptr
                  : graph_->param(name + "_lemmaWt",
                                  {inputDim, lemmaDimEmb},
                                  inits::glorotUniform());    // [D x L] D=hidden-vector dimension
        auto f = lemmaWt ? dot(e, lemmaWt, false, true) : e;  // [B... x D]
        // augment the original hidden vector with this additional information
        input1 = input1 + f;
      }
    }
    return Logits(std::move(allLogits), factoredVocab_);
  } else if(shortlist_) {
    return Logits(affineOrLSH(input,
                              cachedShortWt_,
                              cachedShortb_,
                              false,
                              /*transB=*/isLegacyUntransposedW ? false : true));
  } else {
    return Logits(
        affineOrLSH(input, Wt_, b_, false, /*transB=*/isLegacyUntransposedW ? false : true));
  }
}

}  // namespace mlp
}  // namespace marian