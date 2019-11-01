#include "marian.h"

#include "layers/generic.h"
#include "layers/constructors.h"
#include "layers/loss.h"
#include "data/factored_vocab.h"
#include "rnn/types.h" // for State::select()

//using std::size_t; // not sure why this is needed

namespace marian {
  Logits::Logits(Expr logits) : Logits(New<RationalLoss>(logits, nullptr)) {} // single-output constructor from Expr only (RationalLoss has no count)

  Ptr<ExpressionGraph> Logits::graph() const {
    ABORT_IF(logits_.empty(), "Empty logits object??");
    return logits_.front()->loss()->graph();
  }

  // This function assumes that the object holds one or more factor logits.
  // It applies the supplied loss function to each, and then returns the aggregate loss over all factors.
  Expr Logits::applyLossFunction(const Words& labels, const std::function<Expr(Expr/*logits*/, Expr/*indices*/)>& lossFn) const {
    LOG_ONCE(info, "[logits] applyLossFunction() for {} factors", logits_.size());
    ABORT_IF(empty(), "Attempted to read out logits on empty Logits object");

    auto firstLogits = logits_.front()->loss();
    ABORT_IF(labels.size() * firstLogits->shape()[-1] != firstLogits->shape().elements(), "Labels not matching logits shape??");

    // base case (no factors)
    if (!factoredVocab_) {
      ABORT_IF(logits_.size() != 1, "Factors without factor mappings??");
      return lossFn(firstLogits, indices(toWordIndexVector(labels)));
    }

    auto numGroups = factoredVocab_->getNumGroups();

    // split labels into individual factor labels
    auto allMaskedFactoredLabels = factorizeWords(labels); // [numGroups][labels.size()] = [numGroups][B... flattened]

    //Expr indices = this->indices(toWordIndexVector(labels));
    // accumulate all CEs for all words that have the factor
    // Memory-wise, this is cheap, all temp objects below are batches of scalars or lookup vectors.
    Expr loss;
    for (size_t g = 0; g < numGroups; g++) {
      if (!logits_[g])
        continue; // empty factor  --@TODO: use an array of indices of non-empty logits_[]
      const auto& maskedFactoredLabels = allMaskedFactoredLabels[g]; // array of (word index, mask)
      auto factorIndices = indices (maskedFactoredLabels.indices); // [B... flattened] factor-label indices, or 0 if factor does not apply
      auto factorMask    = constant(maskedFactoredLabels.masks);   // [B... flattened] loss values get multiplied with 0 for labels that don't have this factor
      auto factorLogits  = logits_[g];                             // [B... * Ug] label-wise loss values (not aggregated yet)
      // For each location in [B...] select [indices[B...]]. If not using factor, select [0] and mask it out next.
      auto factorLoss = lossFn(factorLogits->loss(), factorIndices); // [B... x 1]
      factorLoss = factorLoss * reshape(factorMask, factorLoss->shape()); // mask out factor for words that do not have that factor
      loss = loss ? (loss + factorLoss) : factorLoss; // [B... x 1]
    }
    return loss;
  }

  // This function assumes this object holds a single factor that represents a rational loss (with count).
  //Ptr<RationalLoss> Logits::getRationalLoss() const {
  //  ABORT_IF(logits_.size() != 1 || factoredVocab_, "getRationalLoss() cannot be used on multi-factor outputs");
  //  ABORT_IF(!logits_.front()->count(), "getRationalLoss() used on rational loss without count");
  //  return logits_.front();
  //}

  // get logits for one factor group
  // For groupIndex == 0, the function also requires the shortlist if there is one.
  Expr Logits::getFactoredLogits(size_t groupIndex, Ptr<data::Shortlist> shortlist /*= nullptr*/, const std::vector<IndexType>& selIdx /*= {}*/, size_t beamSize /*= 0*/) const {
    ABORT_IF(empty(), "Attempted to read out logits on empty Logits object");
    auto sel = logits_[groupIndex]->loss(); // [localBeamSize, 1, dimBatch, dimFactorVocab]

    // normalize for decoding:
    //  - all secondary factors: subtract their max
    //  - lemma: add all maxes of applicable factors
    if (groupIndex > 0) {
      sel = sel - max(sel, -1);
    }
    else {
      auto numGroups = getNumFactorGroups();
      for (size_t g = 1; g < numGroups; g++) {
        auto factorMaxima = max(logits_[g]->loss(), -1);
        auto factorMasks = constant(getFactorMasks(g, shortlist ? shortlist->indices() : std::vector<WordIndex>()));
        sel = sel + factorMaxima * factorMasks; // those lemmas that don't have a factor get multiplied with 0
      }
    }

    // if selIdx are given, then we must reshuffle accordingly
    if (!selIdx.empty()) // use the same function that shuffles decoder state
      sel = rnn::State::select(sel, selIdx, (int)beamSize, /*isBatchMajor=*/false);
    return sel;
  }

  // used for breakDown() only
  // Index is flattened
  Tensor Logits::getFactoredLogitsTensor(size_t groupIndex) const {
    ABORT_IF(empty(), "Attempted to read out logits on empty Logits object");
    return logits_[groupIndex]->loss()->val();
  }

  // This function assumes that the object holds one or more factor logits, which are summed up
  // into output-vocab logits according to the factored model (with correct normalization of factors).
  // This is infeasible for realistic factor sets, and therefore only implemented for 1 factor.
  // @TODO: remove altogether
  Expr Logits::getLogits() const {
    ABORT_IF(empty(), "Attempted to read out logits on empty Logits object");
    if (!factoredVocab_) {
      ABORT_IF(logits_.size() != 1, "Factors without factor mappings??");
      return getFactoredLogits(0);
    }

#ifdef FACTOR_FULL_EXPANSION
    // compute normalized factor log probs
    std::vector<Expr> logProbs(logits_.size());
    for (size_t g = 0; g < logits_.size(); g++)
      logProbs[g] = logsoftmax(logits_[g]->loss());
    auto y = concatenate(logProbs, /*axis=*/ -1);

    // sum up the unit logits across factors for each target word
    auto graph = y->graph();
    auto factorMatrix = factoredVocab_->getGlobalFactorMatrix(); // [V x U]
    y = dot_csr(
        y,  // [B x U]
        factorMatrix.shape,
        graph->constant({(int)factorMatrix.weights.size()}, inits::fromVector(factorMatrix.weights), Type::float32),
        graph->constant({(int)factorMatrix.indices.size()}, inits::fromVector(factorMatrix.indices), Type::uint32),
        graph->constant({(int)factorMatrix.offsets.size()}, inits::fromVector(factorMatrix.offsets), Type::uint32),
        /*transB=*/ true); // -> [B x V]

    // mask out gaps
    auto gapLogMask = factoredVocab_->getGapLogMask(); // [V]
    y = y + graph->constant({ (int)gapLogMask.size() }, inits::fromVector(gapLogMask), Type::float32);

    return y;
#else
    ABORT("getLogits() no longer supported for actual factored vocab"); // because it is infeasible
#endif
  }

  void Logits::MaskedFactorIndices::push_back(size_t factorIndex) {
    bool isValid = FactoredVocab::isFactorValid(factorIndex);
    indices.push_back(isValid ? (WordIndex)factorIndex : 0);
    masks.push_back((float)isValid);
  }

  std::vector<Logits::MaskedFactorIndices> Logits::factorizeWords(const Words& words) const { // [numGroups][words.size()] -> breaks encoded Word into individual factor indices
    if (!factoredVocab_) {
      ABORT_IF(logits_.size() != 1, "Factors without factor mappings??");
      return {MaskedFactorIndices(words)};
    }
    auto numGroups = factoredVocab_->getNumGroups();
    std::vector<MaskedFactorIndices> res(numGroups);
    for (size_t g = 0; g < numGroups; g++) {
      auto& resg = res[g];
      resg.reserve(words.size());
      for (const auto& word : words)
        resg.push_back(factoredVocab_->getFactor(word, g));
    }
    return res;
  }

  //// use first factor of each word to determine whether it has a specific factor
  //std::vector<float> Logits::getFactorMasks(const Words& words, size_t factorGroup) const { // 1.0 for words that do have this factor; else 0
  //  std::vector<float> res;
  //  res.reserve(words.size());
  //  for (const auto& word : words) {
  //    auto lemma = factoredVocab_->getFactor(word, 0);
  //    res.push_back((float)factoredVocab_->lemmaHasFactorGroup(lemma, factorGroup));
  //  }
  //  return res;
  //}

  // return a vector of 1 or 0 indicating for each lemma whether it has a specific factor
  // If 'indices' is given, then return the masks for the indices; otherwise for all lemmas
  std::vector<float> Logits::getFactorMasks(size_t factorGroup, const std::vector<WordIndex>& indices) const { // [lemmaIndex] -> 1.0 for words that do have this factor; else 0
    size_t n = indices.empty() ? (factoredVocab_->getGroupRange(0).second - factoredVocab_->getGroupRange(0).first) : indices.size();
    std::vector<float> res;
    res.reserve(n);
    // @TODO: we should rearrange lemmaHasFactorGroup as vector[groups[i] of float; then move this into FactoredVocab
    for (size_t i = 0; i < n; i++) {
      auto lemma = indices.empty() ? i : (indices[i] - factoredVocab_->getGroupRange(0).first);
      res.push_back((float)factoredVocab_->lemmaHasFactorGroup(lemma, factorGroup));
    }
    return res;
  }

  Logits Logits::applyUnaryFunction(const std::function<Expr(Expr)>& f) const { // clone this but apply f to all loss values
    std::vector<Ptr<RationalLoss>> newLogits;
    for (const auto& l : logits_)
      newLogits.emplace_back(New<RationalLoss>(f(l->loss()), l->count()));
    return Logits(std::move(newLogits), factoredVocab_);
  }

  Logits Logits::applyUnaryFunctions(const std::function<Expr(Expr)>& f1, const std::function<Expr(Expr)>& fother) const {
      std::vector<Ptr<RationalLoss>> newLogits;
      bool first = true;
      for (const auto& l : logits_) {
        newLogits.emplace_back(New<RationalLoss>((first?f1:fother)(l->loss()), l->count())); // f1 for first, fother for all others
        first = false;
      }
      return Logits(std::move(newLogits), factoredVocab_);
  }

  // @TODO: code dup with above; we can merge it into applyToRationalLoss()
  Logits Logits::withCounts(const Expr& count) const { // create new Logits with 'count' implanted into all logits_
    std::vector<Ptr<RationalLoss>> newLogits;
    for (const auto& l : logits_)
      newLogits.emplace_back(New<RationalLoss>(l->loss(), count));
    return Logits(std::move(newLogits), factoredVocab_);
  }

  namespace mlp {
    /*private*/ void Output::lazyConstruct(int inputDim) {
      // We must construct lazily since we won't know tying nor input dim in constructor.
      if (Wt_)
        return;

      auto name = options_->get<std::string>("prefix");
      auto numOutputClasses = options_->get<int>("dim");

      factoredVocab_ = FactoredVocab::tryCreateAndLoad(options_->get<std::string>("vocab", ""));
      if (factoredVocab_) {
        numOutputClasses = (int)factoredVocab_->factorVocabSize();
        LOG(info, "[embedding] Factored outputs enabled");
      }

      if(tiedParam_) {
        Wt_ = tiedParam_;
      } else {
        if (graph_->get(name + "_W")) { // support of legacy models that did not transpose
          Wt_ = graph_->param(name + "_W", {inputDim, numOutputClasses}, inits::glorotUniform(true, false));
          isLegacyUntransposedW = true;
        }
        else // this is the regular case:
          Wt_ = graph_->param(name + "_Wt", {numOutputClasses, inputDim}, inits::glorotUniform(false, true));
      }

      b_ = graph_->param(name + "_b", {1, numOutputClasses}, inits::zeros());

      /*const*/ int lemmaDimEmb = options_->get<int>("lemma-dim-emb", 0);
      ABORT_IF(lemmaDimEmb && !factoredVocab_, "--lemma-dim-emb requires a factored vocabulary");
      if (lemmaDimEmb > 0) {
#define HARDMAX_HACK
#ifdef HARDMAX_HACK
        lemmaDimEmb = lemmaDimEmb & 0xfffffffe;
#endif
        auto range = factoredVocab_->getGroupRange(0);
        auto lemmaVocabDim = (int)(range.second - range.first);
        auto initFunc = inits::glorotUniform(/*fanIn=*/true, /*fanOut=*/false); // -> embedding vectors have roughly unit length
        lemmaEt_ = graph_->param(name + "_lemmaEt", {lemmaDimEmb, lemmaVocabDim}, initFunc); // [L x U] L=lemmaDimEmb; transposed for speed
      }
    }

    Logits Output::applyAsLogits(Expr input) /*override final*/ {
      lazyConstruct(input->shape()[-1]);

      if (shortlist_ && !cachedShortWt_) { // shortlisted versions of parameters are cached within one batch, then clear()ed
        cachedShortWt_ = index_select(Wt_, isLegacyUntransposedW ? -1 : 0, shortlist_->indices());
        cachedShortb_  = index_select(b_ ,                             -1, shortlist_->indices());
      }

      if (factoredVocab_) {
        auto graph = input->graph();

        // project each factor separately
        auto numGroups = factoredVocab_->getNumGroups();
        std::vector<Ptr<RationalLoss>> allLogits(numGroups, nullptr); // (note: null entries for absent factors)
        Expr input1 = input;
        Expr Plemma = nullptr;
        for (size_t g = 0; g < numGroups; g++) {
          auto range = factoredVocab_->getGroupRange(g);
          if (g > 0 && range.first == range.second) // empty entry
            continue;
          ABORT_IF(g > 0 && range.first != factoredVocab_->getGroupRange(g-1).second, "Factor groups must be consecutive (group {} vs predecessor)", g);
          // slice this group's section out of W_
          Expr factorWt, factorB;
          if (g == 0 && shortlist_) {
            factorWt = cachedShortWt_;
            factorB  = cachedShortb_;
          }
          else {
            factorWt = slice(Wt_, isLegacyUntransposedW ? -1 : 0, Slice((int)range.first, (int)range.second));
            factorB  = slice(b_,                              -1, Slice((int)range.first, (int)range.second));
          }
          // @TODO: b_ should be a vector, not a matrix; but shotlists use cols() in, which requires a matrix
          auto factorLogits = affine(input1, factorWt, factorB, false, /*transB=*/isLegacyUntransposedW ? false : true, /*scale=*/1.0f); // [B... x U] factor logits
          // optionally add lemma-dependent bias
          if (Plemma) { // [B... x U0]
            int lemmaVocabDim = Plemma->shape()[-1];
            int factorVocabDim = factorLogits->shape()[-1];
            auto name = options_->get<std::string>("prefix");
            Expr lemmaBt = graph_->param(name + "_lemmaBt_" + std::to_string(g), {factorVocabDim, lemmaVocabDim}, inits::zeros()); // [U x U0] U0=#lemmas one bias per class per lemma
            auto b = dot(Plemma, lemmaBt, false, true); // [B... x U]
            factorLogits = factorLogits + b;
          }
          allLogits[g] = New<RationalLoss>(factorLogits, nullptr);
          // optionally add a soft embedding of lemma back to create some lemma dependency
          // @TODO: if this works, move it into lazyConstruct
          /*const*/ int lemmaDimEmb = options_->get<int>("lemma-dim-emb", 0);
          if (lemmaDimEmb < 0 && g == 0) {
            ABORT_IF(shortlist_ && lemmaDimEmb != 0, "Lemma-dependent bias with short list is not yet implemented");
            LOG_ONCE(info, "[embedding] using lemma-dependent bias");
            auto factorLogSoftmax = logsoftmax(factorLogits); // (we do that again later, CSE will kick in)
            auto z = /*stopGradient*/(factorLogSoftmax);
            Plemma = exp(z); // [B... x U]
          }
          if (lemmaDimEmb > 0 && g == 0) {
            LOG_ONCE(info, "[embedding] enabled re-embedding of lemma, at dim {}", lemmaDimEmb);
            // compute softmax. We compute logsoftmax() separately because this way, computation will be reused later via CSE
            auto factorLogSoftmax = logsoftmax(factorLogits);
            auto factorSoftmax = exp(factorLogSoftmax);
#ifdef HARDMAX_HACK
            bool hardmax = (lemmaDimEmb & 1) != 0; // odd value triggers hardmax for now (for quick experimentation)
            if (hardmax) {
              lemmaDimEmb = lemmaDimEmb & 0xfffffffe;
              LOG_ONCE(info, "[embedding] HARDMAX_HACK enabled. Actual dim is {}", lemmaDimEmb);
              auto maxVal = max(factorSoftmax, -1);
              factorSoftmax = eq(factorSoftmax, maxVal);
            }
#endif
            // re-embedding lookup, soft-indexed by softmax
            if (shortlist_ && !cachedShortLemmaEt_) // short-listed version of re-embedding matrix
              cachedShortLemmaEt_ = index_select(lemmaEt_, -1, shortlist_->indices());
            auto e = dot(factorSoftmax, cachedShortLemmaEt_ ? cachedShortLemmaEt_ : lemmaEt_, false, true); // [B... x L]
            // project it back to regular hidden dim
            int inputDim = input1->shape()[-1];
            auto name = options_->get<std::string>("prefix");
            // note: if the lemmaEt[:,w] have unit length (var = 1/L), then lemmaWt @ lemmaEt is also length 1
            Expr lemmaWt = inputDim == lemmaDimEmb ? nullptr : graph_->param(name + "_lemmaWt", { inputDim,  lemmaDimEmb }, inits::glorotUniform()); // [D x L] D=hidden-vector dimension
            auto f = lemmaWt ? dot(e, lemmaWt, false, true) : e; // [B... x D]
            // augment the original hidden vector with this additional information
            input1 = input1 + f;
          }
        }
        return Logits(std::move(allLogits), factoredVocab_);
      }
      else if (shortlist_)
        return Logits(affine(input, cachedShortWt_, cachedShortb_, false, /*transB=*/isLegacyUntransposedW ? false : true));
      else
        return Logits(affine(input, Wt_, b_, false, /*transB=*/isLegacyUntransposedW ? false : true));
    }
  }

  Embedding::Embedding(Ptr<ExpressionGraph> graph, Ptr<Options> options) : LayerBase(graph, options) {
    std::string name = opt<std::string>("prefix");
    int dimVoc = opt<int>("dimVocab");
    int dimEmb = opt<int>("dimEmb");

    bool fixed = opt<bool>("fixed", false);

    factoredVocab_ = FactoredVocab::tryCreateAndLoad(options_->get<std::string>("vocab", ""));
    if (factoredVocab_) {
      dimVoc = (int)factoredVocab_->factorVocabSize();
      LOG(info, "[embedding] Factored embeddings enabled");
    }

    // Embedding layer initialization should depend only on embedding size, hence fanIn=false
    auto initFunc = inits::glorotUniform(/*fanIn=*/false, /*fanOut=*/true); // -> embedding vectors have roughly unit length
    
    if (options_->has("embFile")) {
      std::string file = opt<std::string>("embFile");
      if (!file.empty()) {
        bool norm = opt<bool>("normalization", false);
        initFunc = inits::fromWord2vec(file, dimVoc, dimEmb, norm);
      }
    }

    E_ = graph_->param(name, {dimVoc, dimEmb}, initFunc, fixed);
  }

  // helper to embed a sequence of words (given as indices) via factored embeddings
  /*private*/ Expr Embedding::multiRows(const Words& data, float dropProb) const
  {
    auto graph = E_->graph();
    auto factoredData = factoredVocab_->csr_rows(data);
    // multi-hot factor vectors are represented as a sparse CSR matrix
    // [row index = word position index] -> set of factor indices for word at this position
    ABORT_IF(factoredData.shape != Shape({(int)factoredData.offsets.size()-1/*=rows of CSR*/, E_->shape()[0]}), "shape mismatch??");
    // the CSR matrix is passed in pieces
    auto weights = graph->constant({ (int)factoredData.weights.size() }, inits::fromVector(factoredData.weights), Type::float32);
    auto indices = graph->constant({ (int)factoredData.indices.size() }, inits::fromVector(factoredData.indices), Type::uint32);
    auto offsets = graph->constant({ (int)factoredData.offsets.size() }, inits::fromVector(factoredData.offsets), Type::uint32);
    // apply dropout
    // We apply it to the weights, i.e. factors get dropped out separately, but always as entire vectors.
    weights = dropout(weights, dropProb);
    // perform the product
    return csr_dot(factoredData.shape, weights, indices, offsets, E_);
  }

  std::tuple<Expr/*embeddings*/, Expr/*mask*/> Embedding::apply(Ptr<data::SubBatch> subBatch) const /*override final*/ {
    auto graph = E_->graph();
    int dimBatch = (int)subBatch->batchSize();
    int dimEmb = E_->shape()[-1];
    int dimWords = (int)subBatch->batchWidth();

    // factored embeddings:
    //  - regular:
    //     - y = x @ E    x:[B x 1ofV] ; E:[V x D] ; y:[B x D]
    //  - factored:
    //     - u = x @ M    one-hot to U-dimensional multi-hot (all factors in one concatenated space)
    //        - each row of M contains the set of factors for one word => we want a CSR matrix
    //     - y = (x @ M) @ E   (x:[B x 1ofV] ; M:[V x U]) ; E:[U x D] ; y:[B x D]
    //  - first compute x @ M on the CPU
    //     - (Uvalues, Uindices, Uoffsets) = csr_rows(Mvalues, Mindices, Moffsets, subBatch->data()):
    //        - shape (U, specifically) not actually needed here
    //     - foreach input x[i]
    //        - locate row M[i,*]
    //        - copy through its index values (std::vector<push_back>)
    //     - create a matching ones vector (we can keep growing)
    //     - convert to GPU-side CSR matrix. CSR matrix now has #rows equal to len(x)
    //     - CSR matrix product with E
    //     - csr_dot(Uvalues, Uindices, Uoffsets, E_, transposeU)
    //        - double-check if all dimensions are specified. Probably not for transpose (which would be like csc_dot()).
    //  - weighting:
    //     - core factors' gradients are sums over all words that use the factors;
    //        - core factors' embeddings move very fast
    //        - words will need to make up for the move; rare words cannot
    //     - so, we multiply each factor with 1/refCount
    //        - core factors get weighed down a lot
    //        - no impact on gradients, as Adam makes up for it; embeddings still move fast just as before
    //        - but forward pass weighs them down, so that all factors are in a similar numeric range
    //        - if it is required to be in a different range, the embeddings can still learn that, but more slowly

    auto batchEmbeddings = apply(subBatch->data(), {dimWords, dimBatch, dimEmb});
    auto batchMask = graph->constant({dimWords, dimBatch, 1},
                                     inits::fromVector(subBatch->mask()));
    return std::make_tuple(batchEmbeddings, batchMask);
  }

  Expr Embedding::apply(const Words& words, const Shape& shape) const /*override final*/ {
    if (factoredVocab_) {
      Expr selectedEmbs = multiRows(words, options_->get<float>("dropout", 0.0f));        // [(B*W) x E]
      selectedEmbs = reshape(selectedEmbs, shape); // [W, B, E]
      //selectedEmbs = dropout(selectedEmbs, options_->get<float>("dropout", 0.0f), { selectedEmbs->shape()[-3], 1, 1 }); // @TODO: replace with factor dropout
      return selectedEmbs;
    }
    else
      return applyIndices(toWordIndexVector(words), shape);
  }

  Expr Embedding::applyIndices(const std::vector<WordIndex>& embIdx, const Shape& shape) const /*override final*/ {
    ABORT_IF(factoredVocab_, "Embedding: applyIndices must not be used with a factored vocabulary");
    auto selectedEmbs = rows(E_, embIdx);        // [(B*W) x E]
    selectedEmbs = reshape(selectedEmbs, shape); // [W, B, E]
    // @BUGBUG: We should not broadcast along dimBatch=[-2]. Then we can also dropout before reshape() (test that separately)
    selectedEmbs = dropout(selectedEmbs, options_->get<float>("dropout", 0.0f), { selectedEmbs->shape()[-3], 1, 1 });
    return selectedEmbs;
  }

  // standard encoder word embeddings
  /*private*/ Ptr<IEmbeddingLayer> EncoderDecoderLayerBase::createEmbeddingLayer() const {
    auto options = New<Options>(
        "dimVocab", opt<std::vector<int>>("dim-vocabs")[batchIndex_],
        "dimEmb",   opt<int>("dim-emb"),
        "dropout",  dropout_,
        "prefix",   (opt<bool>("tied-embeddings-src") || opt<bool>("tied-embeddings-all")) ? "Wemb" : prefix_ + "_Wemb",
        "fixed",    embeddingFix_,
        "vocab",    opt<std::vector<std::string>>("vocabs")[batchIndex_]); // for factored embeddings
    if(options_->hasAndNotEmpty("embedding-vectors")) {
      auto embFiles = opt<std::vector<std::string>>("embedding-vectors");
      options->set(
          "embFile", embFiles[batchIndex_],
          "normalization", opt<bool>("embedding-normalization"));
    }
    return New<Embedding>(graph_, options);
  }

  // ULR word embeddings
  /*private*/ Ptr<IEmbeddingLayer> EncoderDecoderLayerBase::createULREmbeddingLayer() const {
    return New<ULREmbedding>(graph_, New<Options>(
        "dimSrcVoc",         opt<std::vector<int>>("dim-vocabs")[0],  // ULR multi-lingual src
        "dimTgtVoc",         opt<std::vector<int>>("dim-vocabs")[1],  // ULR monon tgt
        "dimUlrEmb",         opt<int>("ulr-dim-emb"),
        "dimEmb",            opt<int>("dim-emb"),
        "ulr-dropout",       opt<float>("ulr-dropout"),
        "dropout",           dropout_,
        "ulrTrainTransform", opt<bool>("ulr-trainable-transformation"),
        "ulrQueryFile",      opt<std::string>("ulr-query-vectors"),
        "ulrKeysFile",       opt<std::string>("ulr-keys-vectors")));
  }

  // get embedding layer for this encoder or decoder
  // This is lazy mostly because the constructors of the consuming objects are not
  // guaranteed presently to have access to their graph.
  Ptr<IEmbeddingLayer> EncoderDecoderLayerBase::getEmbeddingLayer(bool ulr) const {
    if (embeddingLayers_.size() <= batchIndex_ || !embeddingLayers_[batchIndex_]) { // lazy
      if (embeddingLayers_.size() <= batchIndex_)
        embeddingLayers_.resize(batchIndex_ + 1);
      if (ulr)
        embeddingLayers_[batchIndex_] = createULREmbeddingLayer(); // embedding uses ULR
      else
        embeddingLayers_[batchIndex_] = createEmbeddingLayer();
    }
    return embeddingLayers_[batchIndex_];
  }
}  // namespace marian
