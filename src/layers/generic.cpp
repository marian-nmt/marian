#include "marian.h"

#include "layers/generic.h"

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
    // Note: Presently, this implementation has the following short-comings
    //  - we do not group factors (to normalize the probs); instead we assume that the global softmax will normalize correctly
    //  - we do not handle binary features differently; we'd need to apply sigmoid(x) / sigmoid(-x)
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

      // load and parse factorMap
      factorMap_.resize(vocab.size());
      factorRefCounts_.resize(factorVocab_.size());
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
      LOG(info, "[embedding] Factored-embedding map read with total/unique of {}/{} factors for {} words", numTotalFactors, factorVocab_.size(), vocab.size());

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

    const CSRData& getFactorMatrix() const { return factorMatrix_; }
  private:
    Vocab factorVocab_;                        // [factor name] -> factor index = row of E_
    std::vector<std::vector<Word>> factorMap_; // [word index] -> set of factor indices
    std::vector<int> factorRefCounts_;         // [factor index] -> how often this factor is referenced in factorMap_
    CSRData factorMatrix_;                     // [v,u] (sparse) -> =1 if u is factor of v
  };

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
        transposeW_ = true;
        W_ = tiedParam_;
        if(shortlist_)
          W_ = rows(W_, shortlist_->indices());
      } else {
        W_ = graph_->param(name + "_W",
                           {inputDim, dim},
                           inits::glorot_uniform);
        if(shortlist_)
          W_ = cols(W_, shortlist_->indices());
      }

      b_ = graph_->param(name + "_b", {1, dim}, inits::zeros);
      if(shortlist_)
        b_ = cols(b_, shortlist_->indices());
    }

    Expr Output::apply(Expr input) /*override*/ {
      lazyConstruct(input->shape()[-1]);
      auto y = affine(input, W_, b_, false, transposeW_);
      if (embeddingFactorMapping_) {
        auto graph = input->graph();
        auto factorMatrix = embeddingFactorMapping_->getFactorMatrix(); // [V x U]
        y = dot_csr( // the CSR matrix is passed in pieces
            y,  // [B x U]
            factorMatrix.shape,
            graph->constant({(int)factorMatrix.weights.size()}, inits::from_vector(factorMatrix.weights), Type::float32),
            graph->constant({(int)factorMatrix.indices.size()}, inits::from_vector(factorMatrix.indices), Type::uint32),
            graph->constant({(int)factorMatrix.offsets.size()}, inits::from_vector(factorMatrix.offsets), Type::uint32),
            /*transB=*/ true); // -> [B x V]
      }
      return y;
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
