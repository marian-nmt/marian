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
    Vocab factorVocab_;                        // [factor name] -> factor index = row of E_
    std::vector<std::vector<Word>> factorMap_; // [word index] -> set of factor indices
    std::vector<int> factorRefCounts_;         // [factor index] -> how often this factor is referenced in factorMap_
  public:
    // mapPath = path to file with entries in order of vocab entries of the form
    //   WORD FACTOR1 FACTOR2 FACTOR3...
    // listPath = path to file that lists all FACTOR names
    // vocab = original vocabulary
    // Note: The WORD field in the map file is redundant. It is required for consistency checking only.
    // Note: Presently, this implementation has the following short-comings
    //  - we do not group factors (to normalize the probs); instead we assume that the global softmax will normalize correctly
    //  - we do not handle binary features differently; we'd need to apply sigmoid(x) / sigmoid(-x)
    EmbeddingFactorMapping(const std::string& mapPath, const std::string& factorVocabPath, const std::string& vocabPath) :
        factorVocab_(New<Options>(), 0) {
      // Note: We misuse the Vocab class a little.
      // But it means that the factorVocab_ must contain </s> and "<unk>".
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
    }

    size_t factorVocabSize() const { return factorVocab_.size(); }

    // create a CSR matrix M[V,U] from indices[] with
    // M[v,u] = 1/c(u) if factor u is a factor of word v, and c(u) is how often u is referenced
    std::tuple<Shape, std::vector<float>/*weights*/, std::vector<IndexType>/*indices*/, std::vector<IndexType>/*offsets*/> csr_rows(const std::vector<IndexType>& words) const {
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
      return
          std::tuple<Shape, std::vector<float>/*weights*/, std::vector<IndexType>/*indices*/, std::vector<IndexType>/*offsets*/> // (needed for unknown reasons)
          { Shape({(int)words.size(), (int)factorVocab_.size()}), weights, indices, offsets };
    }
  };

  Embedding::Embedding(Ptr<ExpressionGraph> graph, Ptr<Options> options) : LayerBase(graph, options) {
    std::string name = opt<std::string>("prefix");
    int dimVoc = opt<int>("dimVocab");
    int dimEmb = opt<int>("dimEmb");

    bool fixed = opt<bool>("fixed", false);

    if (options_->has("embedding-factors") && !embeddingFactorMapping_) { // (lazy init)
      std::vector<std::string> paths = opt<std::vector<std::string>>("embedding-factors");
      ABORT_IF(paths.size() != 2, "--embedding-factors expects two paths");
      embeddingFactorMapping_ = New<EmbeddingFactorMapping>(paths[0], paths[1], opt<std::string>("vocab"));
      dimVoc = (int)embeddingFactorMapping_->factorVocabSize();
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
    Shape shape; std::vector<float> weights;  std::vector<IndexType> indices, offsets; std::tie
    (shape, weights, indices, offsets) = embeddingFactorMapping_->csr_rows(data);
#if 0   // tests for special case of nop-op
    ABORT_IF(data != indices, "bang");
    for (size_t i = 0; i < offsets.size(); i++)
      ABORT_IF(offsets[i] != i, "boom");
    for (size_t i = 0; i < weights.size(); i++)
      ABORT_IF(weights[i] != 1, "oops");
#endif
    // multi-hot factor vectors are represented as a sparse CSR matrix
    // [row index = word position index] -> set of factor indices for word at this position
    ABORT_IF(shape != Shape({(int)offsets.size()-1/*=rows of CSR*/, E_->shape()[0]}), "shape mismatch??");
    return csr_dot( // the CSR matrix is passed in pieces
        shape,
        graph->constant({(int)weights.size()}, inits::from_vector(weights), Type::float32),
        graph->constant({(int)indices.size()}, inits::from_vector(indices), Type::uint32),
        graph->constant({(int)offsets.size()}, inits::from_vector(offsets), Type::uint32),
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
