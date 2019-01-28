#include "marian.h"

#include "layers/generic.h"

using std::size_t; // not sure why this is needed

namespace marian {
  struct CSRSparseTensor { // simplistic for now
    Shape shape;
    Expr values;  // [k_i..k_{i+1}-1] -> value at [i,j]
    Expr indices; // [k_i..k_{i+1}-1] -> j of non-null value
    Expr offsets; // [i] -> k_i
  };

  class EmbeddingFactorMapping {
  public:
    struct CSRData {
      Shape shape;
      std::vector<float> weights;
      std::vector<IndexType> indices;
      std::vector<IndexType> offsets;
    };
    // mapPath = path to file with entries in order of vocab entries of the form
    //   WORD FACTOR1 FACTOR2 FACTOR3...
    // listPath = path to file that lists all FACTOR names
    // vocab = original vocabulary
    // Note: The WORD field in the map file is redundant. It is required for consistency checking only.
    // Factors are grouped
    //  - user specifies list-factor prefixes; all factors beginning with that prefix are in the same group
    //  - factors within a group as multi-class and normalized that way
    //  - groups of size 1 are interpreted as sigmoids, multiply with P(u) / P(u-1)
    //  - one prefix must not contain another
    //  - all factors not matching a prefix get lumped into yet another class (the lemmas)
    //  - factor vocab must be sorted such that all groups are consecutive
    //  - result of Output layer is nevertheless logits, not a normalized probability, due to the sigmoid entries
    EmbeddingFactorMapping(Ptr<Options> options) : factorVocab_(New<Options>(), 0) {
      std::vector<std::string> paths = options->get<std::vector<std::string>>("embedding-factors");
      ABORT_IF(paths.size() != 2, "--embedding-factors expects two paths");
      auto mapPath = paths[0];
      auto factorVocabPath = paths[1];
      auto vocabPath = options->get<std::string>("vocab");

      // Note: We misuse the Vocab class a little.
      // Specifically, it means that the factorVocab_ must contain </s> and "<unk>".
      Vocab vocab(New<Options>(), 0);
      vocab.load(vocabPath);
      factorVocab_.load(factorVocabPath);
      Word numFactors = (Word)factorVocab_.size();

      // load and parse factorMap
      factorMap_.resize(vocab.size());
      factorRefCounts_.resize(numFactors);
      std::vector<std::string> tokens;
      io::InputFileStream in(mapPath);
      std::string line;
      size_t numTotalFactors = 0;
      for (Word v = 0; io::getline(in, line); v++) {
        tokens.clear(); // @BUGBUG: should be done in split()
        utils::splitAny(line, tokens, " \t");
        ABORT_IF(tokens.size() < 2 || tokens.front() != vocab[v], "Factor map must list words in same order as vocab, and have at least one factor per word", mapPath);
        for (size_t i = 1; i < tokens.size(); i++) {
          auto u = factorVocab_[tokens[i]];
          auto& m = factorMap_[v];
          m.push_back(u);
          factorRefCounts_[u]++;
        }
        numTotalFactors += tokens.size() - 1;
      }
      LOG(info, "[embedding] Factored-embedding map read with total/unique of {}/{} factors for {} words", numTotalFactors, numFactors, vocab.size());

      // form groups
      // @TODO: hard-coded for these initial experiments
      std::vector<std::string> groupPrefixes = {
        "@C",
        "@GL", "@GR"
      };
      groupPrefixes.insert(groupPrefixes.begin(), "(unassigned)");     // first group is fallback for normal words (the string is only used for messages)
      size_t numGroups = groupPrefixes.size();
      factorGroups_.resize(numFactors, 0);
      for (size_t g = 1; g < groupPrefixes.size(); g++) { // set group labels; what does not match any prefix will stay in group 0
        const auto& groupPrefix = groupPrefixes[g];
        for (Word u = 0; u < numFactors; u++)
          if (utils::beginsWith(factorVocab_[u], groupPrefix)) {
            ABORT_IF(factorGroups_[u] != 0, "Factor {} matches multiple groups, incl. {}", factorVocab_[u], groupPrefix);
            factorGroups_[u] = g;
          }
      }
      groupRanges_.resize(numGroups, { SIZE_MAX, (size_t)0 });
      std::vector<size_t> groupCounts(numGroups); // number of group members
      for (Word u = 0; u < numFactors; u++) { // determine ranges; these must be non-overlapping, verified via groupCounts
        auto g = factorGroups_[u];
        if (groupRanges_[g].first > u)
            groupRanges_[g].first = u;
        if (groupRanges_[g].second < u + 1)
            groupRanges_[g].second = u + 1;
        groupCounts[g]++;
      }
      // create the flag vectors for normalization   --@TODO: maybe we won't need them anymore
      mVecs_.resize(numGroups);
      for (size_t g = 0; g < numGroups; g++) { // detect non-overlapping groups
        LOG(info, "[embedding] Factor group '{}' has {} members ({})",
            groupPrefixes[g], groupCounts[g], groupCounts[g] == 1 ? "sigmoid" : "softmax");
        if (groupCounts[g] == 0) // factor group is unused  --@TODO: once this is not hard-coded, this is an error condition
          continue;
        ABORT_IF(groupRanges_[g].second - groupRanges_[g].first != groupCounts[g],
                 "Factor group '{}' members should be consecutive in the factor vocabulary", groupPrefixes[g]);
        auto& mVec = mVecs_[g];
        mVec.resize(numFactors, 0.0f);
        for (size_t i = groupRanges_[g].first; i < groupRanges_[g].second; i++)
          mVec[i] = 1.0f;
      }

      // create the factor matrix
      std::vector<IndexType> data(vocab.size());
      std::iota(data.begin(), data.end(), 0);
      factorMatrix_ = csr_rows(data); // [V x U]
    }

    size_t factorVocabSize() const { return factorVocab_.size(); }

    // create a CSR matrix M[V,U] from indices[] with
    // M[v,u] = 1/c(u) if factor u is a factor of word v, and c(u) is how often u is referenced
    CSRData csr_rows(const std::vector<IndexType>& words) const {
      std::vector<float> weights;
      std::vector<IndexType> indices;
      std::vector<IndexType> offsets;
      offsets.reserve(words.size() + 1);
      indices.reserve(words.size()); // (at least this many)
      // loop over all input words, and select the corresponding set of unit indices into CSR format
      offsets.push_back((IndexType)indices.size());
      for (auto v : words) {
        const auto& m = factorMap_[v];
        for (auto u : m) {
          indices.push_back(u);
          weights.push_back(1.0f/*/(float)factorRefCounts_[u]*/);
        }
        offsets.push_back((IndexType)indices.size()); // next matrix row begins at this offset
      }
      return { Shape({(int)words.size(), (int)factorVocab_.size()}), weights, indices, offsets };
    }

    const CSRData& getFactorMatrix() const { return factorMatrix_; } // [v,u] (sparse) -> =1 if u is factor of v
  private:
    Vocab factorVocab_;                                  // [factor name] -> factor index = row of E_
    std::vector<std::vector<Word>> factorMap_;           // [word index] -> set of factor indices
    std::vector<int> factorRefCounts_;                   // [factor index] -> how often this factor is referenced in factorMap_
    CSRData factorMatrix_;                               // [v,u] (sparse) -> =1 if u is factor of v
    std::vector<size_t> factorGroups_;                   // [u] -> group id of factor u
  public: // @TODO: temporarily; later factor this properly
    std::vector<std::pair<size_t, size_t>> groupRanges_; // [group id] -> (u_begin,u_end) index range of factors u for this group. These don't overlap.
    std::vector<std::vector<float>> mVecs_;              // [group id][u] -> 1 if factor is member of group
  };

  Expr Logits::getLogits() const {
    ABORT_IF(empty(), "Attempted to read out logits on empty Logits object");
    if (!embeddingFactorMapping_) {
      ABORT_IF(logits_.size() != 1, "Factors without factor mappings??");
      return logits_.front();
    }

    // lazily compute combined logits from factors
    auto y = concatenate(logits_, /*axis=*/ -1);

    // sum up the unit logits across factors for each target word
    auto graph = y->graph();
    auto factorMatrix = embeddingFactorMapping_->getFactorMatrix(); // [V x U]
    y = dot_csr(
        y,  // [B x U]
        factorMatrix.shape,
        graph->constant({(int)factorMatrix.weights.size()}, inits::from_vector(factorMatrix.weights), Type::float32),
        graph->constant({(int)factorMatrix.indices.size()}, inits::from_vector(factorMatrix.indices), Type::uint32),
        graph->constant({(int)factorMatrix.offsets.size()}, inits::from_vector(factorMatrix.offsets), Type::uint32),
        /*transB=*/ true); // -> [B x V]

    return y;
#if 0
        // denominators
        const auto& mVecs = embeddingFactorMapping_->mVecs_;
        for (size_t g = 0; g < numGroups; g++) {
          auto range = groupRanges[g];
          // y: [B... x U]
          // m: [1 x U]         // ones at positions of group members
          // need to compute log denominator over y[range] and subtract it from y[range]
          //auto groupY = slice(y, /*axis=*/-1, Slice((int)range.first, (int)range.second)); // [B... x Ug]
          //auto groupZ = logsumexp(groupY, /*axis=*/-1); // [B... x 1]
          ////auto groupZ = slice(groupY - logsoftmax(groupY), /*axis=*/-1, 0); // [B... x 1]
          const auto& mVec = mVecs[g];
          auto m = graph->constant({ 1, (int)mVec.size() }, inits::from_vector(mVec)); // [1 x U]
          //auto Z = dot(groupZ, m); // [B... x U]
          //y = y - Z;
          // and a log-linear weight
          auto name = options_->get<std::string>("prefix");
          auto groupLLWeights[g] = graph->param(name + "_llWeight_" + std::to_string(g), {}, inits::from_value(1.0f));
          y = y * ((groupLLWeights[g] - 1) * m + 1);
          // @BUGBUG: Global softmax no longer normalizes, due to words that lack some factors.
        }
#endif
  }

  namespace mlp {
    /*private*/ void Output::lazyConstruct(int inputDim) {
      // We must construct lazily since we won't know tying nor input dim in constructor.
      if (W_)
        return;

      auto name = options_->get<std::string>("prefix");
      auto dim = options_->get<int>("dim");

      if (options_->has("embedding-factors")) {
        ABORT_IF(shortlist_, "Shortlists are presently not compatible with factored embeddings");
        embeddingFactorMapping_ = New<EmbeddingFactorMapping>(options_);
        dim = (int)embeddingFactorMapping_->factorVocabSize();
        LOG(info, "[embedding] Factored outputs enabled");
      }

      if(tiedParam_) {
        W_ = tiedParam_;
        transposeW_ = true;
      } else {
        W_ = graph_->param(name + "_W", {inputDim, dim}, inits::glorot_uniform);
        transposeW_ = false;
      }

      b_ = graph_->param(name + "_b", {1, dim}, inits::zeros);
    }

    Logits Output::apply(Expr input) /*override*/ {
      lazyConstruct(input->shape()[-1]);

      if (shortlist_) {
        if (!cachedShortW_) { // short versions of parameters are cached within one batch, then clear()ed
          if(transposeW_)
            cachedShortW_ = rows(W_, shortlist_->indices());
          else
            cachedShortW_ = cols(W_, shortlist_->indices());
          cachedShortb_ = cols(b_, shortlist_->indices());
        }
        return affine(input, cachedShortW_, cachedShortb_, false, transposeW_);
      }
      else if (embeddingFactorMapping_) {
        auto graph = input->graph();

        // project each factor
        const auto& groupRanges = embeddingFactorMapping_->groupRanges_; // @TODO: factor this properly
        auto numGroups = groupRanges.size();
        std::vector<Expr> groupYs(numGroups);
        std::vector<Expr> groupLLWeights(numGroups);
        for (size_t g = 0; g < numGroups; g++) {
          auto range = groupRanges[g];
          ABORT_IF(g > 0 && groupRanges[g].first != groupRanges[g-1].second, "Factor groups must be consecutive"); // we could sort groupYs though
          // slice this group's section out of W_
          // @TODO: This is highly inefficient if not tied. We should always transpose Output's matrix.
          auto groupW = slice(W_, transposeW_ ? 0 : -1, Slice((int)range.first, (int)range.second));
          //LOG(info, "slice() -> {}, {}", groupW->type(), std::string(groupW->shape()));
          auto groupB = slice(b_, -1, Slice((int)range.first, (int)range.second)); // @TODO: b_ should be a vector, not a matrix
          auto groupY = affine(input, groupW, groupB, false, transposeW_); // [B... x U] factor logits
          // normalize
          groupY = logsoftmax(groupY);
          // log-linear weight   --@TODO: pre-create in constructor
          auto name = options_->get<std::string>("prefix");
          groupLLWeights[g] = graph->param(name + "_llWeight_" + std::to_string(g), {}, inits::from_value(1.0f));
          groupY = groupY * groupLLWeights[g];
          // @BUGBUG: Global softmax no longer normalizes, due to words that lack some factors.
          // @TODO: normalize again. Do I need the first normalization?
          groupYs[g] = groupY;
        }
        return Logits(std::move(groupYs), embeddingFactorMapping_);
      }
      else
        return affine(input, W_, b_, false, transposeW_);
    }
  }

  Embedding::Embedding(Ptr<ExpressionGraph> graph, Ptr<Options> options) : LayerBase(graph, options) {
    std::string name = opt<std::string>("prefix");
    int dimVoc = opt<int>("dimVocab");
    int dimEmb = opt<int>("dimEmb");

    bool fixed = opt<bool>("fixed", false);

    if (options_->has("embedding-factors")) {
      embeddingFactorMapping_ = New<EmbeddingFactorMapping>(options_);
      dimVoc = (int)embeddingFactorMapping_->factorVocabSize();
      LOG(info, "[embedding] Factored embeddings enabled");
    }

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
    auto factoredData = embeddingFactorMapping_->csr_rows(data);
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

    Expr chosenEmbeddings;
    if (embeddingFactorMapping_)
      chosenEmbeddings = multiRows(subBatch->data());
    else
      chosenEmbeddings = rows(E_, subBatch->data());

    auto batchEmbeddings = reshape(chosenEmbeddings, { dimWords, dimBatch, dimEmb });
    auto batchMask = graph->constant({ dimWords, dimBatch, 1 },
                                     inits::from_vector(subBatch->mask()));
    return std::make_tuple(batchEmbeddings, batchMask);
  }

  Expr Embedding::apply(const std::vector<IndexType>& embIdx, int dimBatch, int dimBeam) const /*override final*/ {
    int dimEmb = E_->shape()[-1];
    Expr chosenEmbeddings;
    if (embeddingFactorMapping_)
      chosenEmbeddings = multiRows(embIdx);
    else
      chosenEmbeddings = rows(E_, embIdx);
    return reshape(chosenEmbeddings, { dimBeam, 1, dimBatch, dimEmb });
  }
}  // namespace marian
