#include "output.h"
#include "common/timer.h"
#include "data/factored_vocab.h"
#include "layers/loss.h"

namespace marian {
namespace mlp {

/*private*/ void Output::lazyConstruct(int inputDim) {
  // We must construct lazily since we won't know tying nor input dim in constructor.
  if(Wt_)
    return;

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
  std::string lemmaDependency = options_->get<std::string>("lemma-dependency", "");
  ABORT_IF(lemmaDimEmb && !factoredVocab_, "--lemma-dim-emb requires a factored vocabulary");
  if(lemmaDependency == "re-embedding") {  // embed the (expected) word with a different embedding matrix
    ABORT_IF(
        lemmaDimEmb <= 0,
        "In order to predict factors by re-embedding them, a lemma-dim-emb must be specified.");
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
    /*
    std::cerr << "affineOrDot.x=" << x->shape() << std::endl;
    std::cerr << "affineOrDot.W=" << W->shape() << std::endl;
    if (b) std::cerr << "affineShortlist.b=" << b->shape() << std::endl;
    std::cerr << "affineOrDot.transA=" << transA << " transB=" << transB << std::endl;
    */
    if(b)
      return affine(x, W, b, transA, transB);
    else
      return dot(x, W, transA, transB);
  };

  auto affineShortlist = [this](Expr x, Expr W, Expr b, bool transA, bool transB) {
    /*    
    std::cerr << "affineShortlist.x=" << x->shape() << std::endl;
    std::cerr << "affineShortlist.W=" << W->shape() << std::endl;
    if (b) std::cerr << "affineShortlist.b=" << b->shape() << std::endl;
    std::cerr << "affineShortlist.transA=" << transA << " transB=" << transB << std::endl;
    */

    Expr ret;

    if (b) {
      // original shortlist. W always has 1 for beam & batch
      ABORT_UNLESS(!shortlist_->isDynamic(), "affineShortlist. Bias not supported with LSH/dynamic shortlist"); // todo rename ABORT_UNLESS to ASSERT
      ret = affine(x, W, b, transA, transB);
    }
    else if (shortlist_->isDynamic()) {
      // LSH produces W entry for each beam and batch => need bdot()
      ABORT_IF(!(!transA && transB), "affineShortlist. Only tested with transA==0 and transB==1");
      ret = bdot(x, W, transA, transB);
    }
    else {
      // original shortlist. W always has 1 for beam & batch
      ret = dot(x, W, transA, transB);
    } 

    //std::cerr << "ret.x=" << ret->shape() << std::endl;
    return ret;
  };

  if(shortlist_) {
    shortlist_->filter(input, Wt_, isLegacyUntransposedW, b_, lemmaEt_);
  }

  if(factoredVocab_) {
    auto graph = input->graph();

    // project each factor separately
    auto numGroups = factoredVocab_->getNumGroups();
    std::vector<Ptr<RationalLoss>> allLogits(numGroups,
                                             nullptr);  // (note: null entries for absent factors)
    Expr input1 = input;                                // [B... x D]
    Expr Plemma = nullptr;                              // used for lemmaDependency = lemma-dependent-bias
    Expr inputLemma = nullptr;                          // used for lemmaDependency = hard-transformer-layer and soft-transformer-layer

    std::string factorsCombine = options_->get<std::string>("factors-combine", "");
    ABORT_IF(factorsCombine == "concat", "Combining lemma and factors embeddings with concatenation on the target side is currently not supported");

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
        factorWt = shortlist_->getCachedShortWt();
        factorB = shortlist_->getCachedShortb();
      } else {
        factorWt = slice(
            Wt_, isLegacyUntransposedW ? -1 : 0, Slice((int)range.first, (int)range.second));
        if(hasBias_)
          factorB = slice(b_, -1, Slice((int)range.first, (int)range.second));
      }
      /*const*/ int lemmaDimEmb = options_->get<int>("lemma-dim-emb", 0);
      std::string lemmaDependency = options_->get<std::string>("lemma-dependency", "");
      if((lemmaDependency == "soft-transformer-layer" || lemmaDependency == "hard-transformer-layer") && g > 0) {
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
                             "relu",
                             ffnDropProb);
        f = denseInline(f, name + "_ffn", /*suffix=*/"2", inputDim);
        // add & norm
        input1 = f + input1;
        input1 = layerNorm(input1, name + "_ffn");
      }
      // @TODO: b_ should be a vector, not a matrix; but shotlists use cols() in, which requires a
      // matrix
      Expr factorLogits;
      if(g == 0 && shortlist_) {
        Expr tmp = transpose(input1, {0, 2, 1, 3});
        factorLogits = affineShortlist(
            tmp,
            factorWt,
            factorB,
            false,
            /*transB=*/isLegacyUntransposedW ? false : true);  // [B... x U] factor logits
        factorLogits = transpose(factorLogits, {0, 2, 1, 3});
      }
      else {
        factorLogits = affineOrDot(
            input1,
            factorWt,
            factorB,
            false,
            /*transB=*/isLegacyUntransposedW ? false : true);  // [B... x U] factor logits
      }

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
      //std::cerr << "factorLogits=" << factorLogits->shape() << std::endl;
      allLogits[g] = New<RationalLoss>(factorLogits, nullptr);
      // optionally add a soft embedding of lemma back to create some lemma dependency
      // @TODO: if this works, move it into lazyConstruct
      if(lemmaDependency == "soft-transformer-layer" && g == 0) {
        LOG_ONCE(info, "[embedding] using lemma conditioning with gate, soft-max version");
        // get expected lemma embedding vector
        auto factorLogSoftmax = logsoftmax(
            factorLogits);  // [B... x U] note: with shortlist, this is not the full lemma set
        auto factorSoftmax = exp(factorLogSoftmax);
        inputLemma = dot(factorSoftmax,
                         factorWt,
                         false,
                         /*transB=*/isLegacyUntransposedW ? true : false);  // [B... x D]
      } else if(lemmaDependency == "hard-transformer-layer" && g == 0) {
        LOG_ONCE(info, "[embedding] using lemma conditioning with gate, hard-max version");
        // get max-lemma embedding vector
        auto maxVal = max(factorLogits,
                          -1);  // [B... x U] note: with shortlist, this is not the full lemma set
        auto factorHardmax = eq(factorLogits, maxVal);
        inputLemma = dot(factorHardmax,
                         factorWt,
                         false,
                         /*transB=*/isLegacyUntransposedW ? true : false);  // [B... x D]
      } else if(lemmaDependency == "lemma-dependent-bias" && g == 0) {
        ABORT_IF(shortlist_, "Lemma-dependent bias with short list is not yet implemented");
        LOG_ONCE(info, "[embedding] using lemma-dependent bias");
        auto factorLogSoftmax
            = logsoftmax(factorLogits);  // (we do that again later, CSE will kick in)
        auto z = /*stopGradient*/ (factorLogSoftmax);
        Plemma = exp(z);                      // [B... x U]
      } else if(lemmaDependency == "re-embedding" && g == 0) {
        ABORT_IF(
            lemmaDimEmb <= 0,
            "In order to predict factors by re-embedding them, a lemma-dim-emb must be specified.");
        LOG_ONCE(info, "[embedding] enabled re-embedding of lemma, at dim {}", lemmaDimEmb);
        // compute softmax. We compute logsoftmax() separately because this way, computation will be
        // reused later via CSE
        auto factorLogSoftmax = logsoftmax(factorLogits);
        auto factorSoftmax = exp(factorLogSoftmax);
        // re-embedding lookup, soft-indexed by softmax
        Expr e;
        if(shortlist_) {  // short-listed version of re-embedding matrix
          Expr cachedShortLemmaEt = shortlist_->getCachedShortLemmaEt();
          // std::cerr << "factorSoftmax=" << factorSoftmax->shape() << std::endl;
          // std::cerr << "cachedShortLemmaEt=" << cachedShortLemmaEt->shape() << std::endl;
          const Shape &fShape = factorSoftmax->shape();
          ABORT_IF(fShape[1] != 1, "We are decoding with a shortlist but time step size {} != 1??", fShape[1]);
          factorSoftmax = reshape(factorSoftmax, {fShape[0], fShape[2], 1, fShape[3]}); // we can switch dims because time step is of size 1
          // std::cerr << "factorSoftmax=" << factorSoftmax->shape() << std::endl;
          e = bdot(factorSoftmax, cachedShortLemmaEt, false, true);
          // std::cerr << "e.1=" << e->shape() << std::endl;
          const Shape &eShape = e->shape();
          e = reshape(e, {eShape[0], 1, eShape[1], eShape[3]}); // switch dims back, again possible because time step is of size 1
          // std::cerr << "e.2=" << e->shape() << std::endl;
          // std::cerr << std::endl;
        } else { // for scoring, training and decoding without a shortlist we use a simple dot operation
          e = dot(factorSoftmax,
                  lemmaEt_,
                  false,
                  true);  // [B... x L]
        }

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
    const Shape &inputShape = input->shape();
    assert(inputShape[1] == 1); // time dimension always 1 for decoding
    input = reshape(input, {inputShape[0], inputShape[2], 1, inputShape[3]});

    Expr Wt = shortlist_->getCachedShortWt();
    Expr b = shortlist_->getCachedShortb();
    Expr ret = affineShortlist(input,
                              Wt,
                              b,
                              false,
                              /*transB=*/isLegacyUntransposedW ? false : true);
    const Shape &retShape = ret->shape();
    assert(retShape[2] == 1); // time dimension always 1 for decoding
    ret = reshape(ret, {retShape[0], 1, retShape[1], retShape[3]});
    return Logits(ret);
  } else {
    Expr ret = affineOrDot(input, Wt_, b_, false, /*transB=*/isLegacyUntransposedW ? false : true);
    return Logits(ret);
  }
}

}  // namespace mlp
}  // namespace marian