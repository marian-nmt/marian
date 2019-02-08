#include "marian.h"

#include "layers/generic.h"
#include "layers/loss.h"
#include "data/factored_vocab.h"

using std::size_t; // not sure why this is needed

namespace marian {
  //struct CSRSparseTensor { // simplistic for now
  //  Shape shape;
  //  Expr values;  // [k_i..k_{i+1}-1] -> value at [i,j]
  //  Expr indices; // [k_i..k_{i+1}-1] -> j of non-null value
  //  Expr offsets; // [i] -> k_i
  //};

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
      const auto& maskedFactoredLabels = allMaskedFactoredLabels[g]; // array of (word index, mask)
#if 1
      auto factorIndices = indices (maskedFactoredLabels.indices); // [B... flattened] factor-label indices, or 0 if factor does not apply
      auto factorMask    = constant(maskedFactoredLabels.masks);   // [B... flattened] loss values get multiplied with 0 for labels that don't have this factor
#else // @TODO: if this^^ works, we can remove the stuff below (quite a bit code)
      maskedFactoredLabels;
      indices; // [B... * 1] all batch items flattened
      auto factorMaskVector  = factoredVocab_->getFactorMasks(g);   // [v] 1.0 if v has factor of group g
      auto factorIndexVector = factoredVocab_->getFactorIndices(g); // [v] index of factor for word v in group p; must be 0 if factor is not used
      auto factorMaskMatrix  = constant({(int)factorMaskVector.size(),  1}, factorMaskVector);  // [V x 1]
      auto factorIndexMatrix = constant({(int)factorIndexVector.size(), 1}, factorIndexVector); // [V x 1(Ug)]
      auto factorIndices = rows(factorIndexMatrix, indices); // [B... * 1(Ug)] map word indices to factor indices (indices into factorLogits)
      auto factorMask    = rows(factorMaskMatrix,  indices); // [B... * 1]     flag whether word has the factor in the first place
#endif
      auto factorLogits = logits_[g];                       // [B... * Ug] label-wise loss values (not aggregated yet)
      // For each location in [B...] select [indices[B...]]. If not using factor, select [0] and mask it out next.
      auto factorLoss = lossFn(factorLogits->loss(), factorIndices); // [B... x 1]
      factorLoss = factorLoss * reshape(factorMask, factorLoss->shape()); // mask out factor for words that do not have that factor
      loss = loss ? (loss + factorLoss) : factorLoss; // [B... x 1]
    }
    return loss;
  }

  // This function assumes this object holds a single factor that represents a rational loss (with count).
  Ptr<RationalLoss> Logits::getRationalLoss() const {
    ABORT_IF(logits_.size() != 1 || factoredVocab_, "getRationalLoss() cannot be used on multi-factor outputs");
    ABORT_IF(!logits_.front()->count(), "getRationalLoss() used on rational loss without count");
    return logits_.front();
  }

  // get logits for one factor group
  Expr Logits::getFactoredLogits(size_t groupIndex) const {
    ABORT_IF(empty(), "Attempted to read out logits on empty Logits object");
    return logits_[groupIndex]->loss();
  }

  // This function assumes that the object holds one or more factor logits, which are summed up
  // into output-vocab logits according to the factored model (with correct normalization of factors).
  Expr Logits::getLogits() const {
    ABORT_IF(empty(), "Attempted to read out logits on empty Logits object");
    if (!factoredVocab_) {
      ABORT_IF(logits_.size() != 1, "Factors without factor mappings??");
      return getFactoredLogits(0);
    }

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
        graph->constant({(int)factorMatrix.weights.size()}, inits::from_vector(factorMatrix.weights), Type::float32),
        graph->constant({(int)factorMatrix.indices.size()}, inits::from_vector(factorMatrix.indices), Type::uint32),
        graph->constant({(int)factorMatrix.offsets.size()}, inits::from_vector(factorMatrix.offsets), Type::uint32),
        /*transB=*/ true); // -> [B x V]

    // mask out gaps
    auto gapLogMask = factoredVocab_->getGapLogMask(); // [V]
    y = y + graph->constant({ (int)gapLogMask.size() }, inits::from_vector(gapLogMask), Type::float32);

    return y;
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
        ABORT_IF(shortlist_, "Shortlists are presently not compatible with factored embeddings");
        numOutputClasses = (int)factoredVocab_->factorVocabSize();
        LOG(info, "[embedding] Factored outputs enabled");
      }

      if(tiedParam_) {
        Wt_ = tiedParam_;
      } else {
        if (graph_->get(name + "_W")) { // support of legacy models that did not transpose
          Wt_ = graph_->param(name + "_W", {inputDim, numOutputClasses}, inits::glorot_uniform);
          isLegacyUntransposedW = true;
        }
        else // this is the regular case:
          Wt_ = graph_->param(name + "_Wt", {numOutputClasses, inputDim}, inits::glorot_uniform);
      }

      b_ = graph_->param(name + "_b", {1, numOutputClasses}, inits::zeros);
    }

    Logits Output::apply(Expr input) /*override*/ {
      lazyConstruct(input->shape()[-1]);

      if (shortlist_) {
        if (!cachedShortWt_) { // short versions of parameters are cached within one batch, then clear()ed
          cachedShortWt_ = index_select(Wt_, isLegacyUntransposedW ? -1 : 0, shortlist_->indices());
          cachedShortb_  = index_select(b_ ,                             -1, shortlist_->indices());
        }
        return affine(input, cachedShortWt_, cachedShortb_, false, /*transB=*/isLegacyUntransposedW ? false : true);
      }
      else if (factoredVocab_) {
        auto graph = input->graph();

        // project each factor separately
        auto numGroups = factoredVocab_->getNumGroups();
        std::vector<Ptr<RationalLoss>> allLogits(numGroups);
        for (size_t g = 0; g < numGroups; g++) {
          auto range = factoredVocab_->getGroupRange(g);
          ABORT_IF(g > 0 && range.first != factoredVocab_->getGroupRange(g-1).second, "Factor groups must be consecutive"); // we could sort groupYs though
          // slice this group's section out of W_
          // @TODO: This is highly inefficient if not tied. We should always transpose Output's matrix.
          auto factorWt = slice(Wt_, isLegacyUntransposedW ? -1 : 0, Slice((int)range.first, (int)range.second));
          auto factorB  = slice(b_,                              -1, Slice((int)range.first, (int)range.second));
          // @TODO: b_ should be a vector, not a matrix; but shotlists use cols() in, which requires a matrix
          auto factorLogits = affine(input, factorWt, factorB, false, /*transB=*/isLegacyUntransposedW ? false : true); // [B... x U] factor logits
          allLogits[g] = New<RationalLoss>(factorLogits, nullptr);
        }
        return Logits(std::move(allLogits), factoredVocab_);
      }
      else
        return affine(input, Wt_, b_, false, /*transB=*/isLegacyUntransposedW ? false : true);
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
    //NodeInitializer initFunc = inits::glorot_uniform2(/*fanIn=*/false, /*fanOut=*/true);
    NodeInitializer initFunc = inits::glorot_uniform;
    if (options_->has("embFile")) {
      std::string file = opt<std::string>("embFile");
      if (!file.empty()) {
        bool norm = opt<bool>("normalization", false);
        initFunc = inits::from_word2vec(file, dimVoc, dimEmb, norm);
      }
    }

    E_ = graph_->param(name, {dimVoc, dimEmb}, initFunc, fixed);
  }

  // helper to embed a sequence of words (given as indices) via factored embeddings
  /*private*/ Expr Embedding::multiRows(const std::vector<IndexType>& data) const
  {
    auto graph = E_->graph();
    auto factoredData = factoredVocab_->csr_rows(data);
    // multi-hot factor vectors are represented as a sparse CSR matrix
    // [row index = word position index] -> set of factor indices for word at this position
    ABORT_IF(factoredData.shape != Shape({(int)factoredData.offsets.size()-1/*=rows of CSR*/, E_->shape()[0]}), "shape mismatch??");
    return csr_dot( // the CSR matrix is passed in pieces
        factoredData.shape,
        graph->constant({(int)factoredData.weights.size()}, inits::from_vector(factoredData.weights), Type::float32),
        graph->constant({(int)factoredData.indices.size()}, inits::from_vector(factoredData.indices), Type::uint32),
        graph->constant({(int)factoredData.offsets.size()}, inits::from_vector(factoredData.offsets), Type::uint32),
        E_);
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

#if 1
    auto batchEmbeddings = apply(subBatch->data(), {dimWords, dimBatch, dimEmb});
#else
    Expr selectedEmbs;
    if (factoredVocab_)
      selectedEmbs = multiRows(subBatch->data());
    else
      selectedEmbs = rows(E_, subBatch->data());
    auto batchEmbeddings = reshape(selectedEmbs, { dimWords, dimBatch, dimEmb });
#endif
    auto batchMask = graph->constant({dimWords, dimBatch, 1},
                                     inits::from_vector(subBatch->mask()));
    return std::make_tuple(batchEmbeddings, batchMask);
  }

  Expr Embedding::apply(const Words& words, const Shape& shape) const /*override final*/ {
    return applyIndices(toWordIndexVector(words), shape);
  }

  Expr Embedding::applyIndices(const std::vector<WordIndex>& embIdx, const Shape& shape) const /*override final*/ {
    Expr selectedEmbs;
    if (factoredVocab_)
      selectedEmbs = multiRows(embIdx);
    else
      selectedEmbs = rows(E_, embIdx);
    return reshape(selectedEmbs, shape);
  }
}  // namespace marian
