#include "embedding.h"
#include "data/factored_vocab.h"

namespace marian {

Embedding::Embedding(Ptr<ExpressionGraph> graph, Ptr<Options> options)
    : LayerBase(graph, options), inference_(opt<bool>("inference")) {
  std::string name = opt<std::string>("prefix");
  int dimVoc       = opt<int>("dimVocab");
  int dimEmb       = opt<int>("dimEmb");
  int dimFactorEmb = opt<int>("dimFactorEmb");

  bool fixed = opt<bool>("fixed", false);

  // Embedding layer initialization should depend only on embedding size, hence fanIn=false
  auto initFunc = inits::glorotUniform(
      /*fanIn=*/false, /*fanOut=*/true);  // -> embedding vectors have roughly unit length

  factoredVocab_ = FactoredVocab::tryCreateAndLoad(options_->get<std::string>("vocab", ""));
  if(factoredVocab_) {
    dimVoc = (int)factoredVocab_->factorVocabSize();
    LOG_ONCE(info, "[embedding] Factored embeddings enabled");
    if(opt<std::string>("factorsCombine") == "concat") {
      ABORT_IF(dimFactorEmb == 0,
               "Embedding: If concatenation is chosen to combine the factor embeddings, a factor "
               "embedding size must be specified.");
      int numberOfFactors = (int)factoredVocab_->getTotalFactorCount();
      dimVoc -= numberOfFactors;
      FactorEmbMatrix_
          = graph_->param("factor_" + name, {numberOfFactors, dimFactorEmb}, initFunc, fixed);
      LOG_ONCE(info,
               "[embedding] Combining lemma and factors embeddings with concatenation enabled");
    }
  }

  if(options_->has("embFile")) {
    std::string file = opt<std::string>("embFile");
    if(!file.empty()) {
      bool norm = opt<bool>("normalization", false);
      initFunc  = inits::fromWord2vec(file, dimVoc, dimEmb, norm);
    }
  }

  E_ = graph_->param(name, {dimVoc, dimEmb}, initFunc, fixed);
}

/**
 * Embeds a sequence of words (given as indices), where they have factor information. The matrices are concatenated
 * @param words vector of words
 * @returns  Expression that is the concatenation of the lemma and factor embeddings
 */
/*private*/ Expr Embedding::embedWithConcat(const Words& data) const {
  auto graph = E_->graph();
  std::vector<IndexType> lemmaIndices;
  std::vector<float> factorIndices;
  factoredVocab_->lemmaAndFactorsIndexes(data, lemmaIndices, factorIndices);
  auto lemmaEmbs = rows(E_, lemmaIndices);
  int dimFactors = FactorEmbMatrix_->shape()[0];
  auto factEmbs
      = dot(graph->constant(
                {(int)data.size(), dimFactors}, inits::fromVector(factorIndices), Type::float32),
            FactorEmbMatrix_);

  return concatenate({lemmaEmbs, factEmbs}, -1);
}

// helper to embed a sequence of words (given as indices) via factored embeddings
Expr Embedding::multiRows(const Words& data, float dropProb) const {
  auto graph        = E_->graph();
  auto factoredData = factoredVocab_->csr_rows(data);
  // multi-hot factor vectors are represented as a sparse CSR matrix
  // [row index = word position index] -> set of factor indices for word at this position
  ABORT_IF(factoredData.shape
               != Shape({(int)factoredData.offsets.size() - 1 /*=rows of CSR*/, E_->shape()[0]}),
           "shape mismatch??");
  // the CSR matrix is passed in pieces
  auto weights = graph->constant({(int)factoredData.weights.size()},
                                 inits::fromVector(factoredData.weights));
  auto indices = graph->constant(
      {(int)factoredData.indices.size()}, inits::fromVector(factoredData.indices), Type::uint32);
  auto offsets = graph->constant(
      {(int)factoredData.offsets.size()}, inits::fromVector(factoredData.offsets), Type::uint32);
  // apply dropout
  // We apply it to the weights, i.e. factors get dropped out separately, but always as entire
  // vectors.
  if(!inference_)
    weights = dropout(weights, dropProb);
  // perform the product
  return csr_dot(factoredData.shape, weights, indices, offsets, E_);
}

std::tuple<Expr /*embeddings*/, Expr /*mask*/> Embedding::apply(Ptr<data::SubBatch> subBatch) const
/*override final*/ {
  auto graph   = E_->graph();
  int dimBatch = (int)subBatch->batchSize();
  int dimEmb   = (factoredVocab_ && opt<std::string>("factorsCombine") == "concat")
                   ? E_->shape()[-1] + FactorEmbMatrix_->shape()[-1]
                   : E_->shape()[-1];
  int dimWidth = (int)subBatch->batchWidth();

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
  //        - double-check if all dimensions are specified. Probably not for transpose (which would
  //        be like csc_dot()).
  //  - weighting:
  //     - core factors' gradients are sums over all words that use the factors;
  //        - core factors' embeddings move very fast
  //        - words will need to make up for the move; rare words cannot
  //     - so, we multiply each factor with 1/refCount
  //        - core factors get weighed down a lot
  //        - no impact on gradients, as Adam makes up for it; embeddings still move fast just as
  //        before
  //        - but forward pass weighs them down, so that all factors are in a similar numeric range
  //        - if it is required to be in a different range, the embeddings can still learn that, but
  //        more slowly

  auto batchEmbeddings = apply(subBatch->data(), {dimWidth, dimBatch, dimEmb});

  auto batchMask = graph->constant({dimWidth, dimBatch, 1}, inits::fromVector(subBatch->mask()));
  // give the graph inputs readable names for debugging and ONNX
  batchMask->set_name("data_" + std::to_string(/*batchIndex_=*/0) + "_mask");

  return std::make_tuple(batchEmbeddings, batchMask);
}

Expr Embedding::apply(const Words& words, const Shape& shape) const /*override final*/ {
  if(factoredVocab_) {
    Expr selectedEmbs;
    if(opt<std::string>("factorsCombine") == "concat")
      selectedEmbs = embedWithConcat(words);  // [(B*W) x E]
    else
      selectedEmbs = multiRows(words, options_->get<float>("dropout", 0.0f));  // [(B*W) x E]
    selectedEmbs = reshape(selectedEmbs, shape);                               // [W, B, E]
    // selectedEmbs = dropout(selectedEmbs, options_->get<float>("dropout", 0.0f), {
    // selectedEmbs->shape()[-3], 1, 1 }); // @TODO: replace with factor dropout
    return selectedEmbs;
  } else
    return applyIndices(toWordIndexVector(words), shape);
}

Expr Embedding::applyIndices(const std::vector<WordIndex>& embIdx, const Shape& shape) const
/*override final*/ {
  ABORT_IF(factoredVocab_, "Embedding: applyIndices must not be used with a factored vocabulary");
  auto embIdxExpr = E_->graph()->indices(embIdx);
  embIdxExpr->set_name("data_"
                       + std::to_string(/*batchIndex_=*/0));  // @TODO: how to know the batch index?
  auto selectedEmbs = rows(E_, embIdxExpr);                   // [(B*W) x E]
  selectedEmbs      = reshape(selectedEmbs, shape);           // [W, B, E]
  // @BUGBUG: We should not broadcast along dimBatch=[-2]. Then we can also dropout before reshape()
  // (test that separately)
  if(!inference_)
    selectedEmbs = dropout(
        selectedEmbs, options_->get<float>("dropout", 0.0f), {selectedEmbs->shape()[-3], 1, 1});
  return selectedEmbs;
}

// standard encoder word embeddings
/*private*/ Ptr<IEmbeddingLayer> EncoderDecoderLayerBase::createEmbeddingLayer() const {
  // clang-format off
  auto options = New<Options>(
      "dimVocab",       opt<std::vector<int>>("dim-vocabs")[batchIndex_],
      "dimEmb",         opt<int>("dim-emb"),
      "dropout",        dropoutEmbeddings_,
      "inference",      inference_,
      "prefix",         (opt<bool>("tied-embeddings-src") || opt<bool>("tied-embeddings-all")) ? "Wemb"
                                                                                          : prefix_ + "_Wemb",
      "fixed",          embeddingFix_,
      "dimFactorEmb",   opt<int>("factors-dim-emb"),  // for factored embeddings
      "factorsCombine", opt<std::string>("factors-combine"),  // for factored embeddings
      "vocab",     opt<std::vector<std::string>>("vocabs")[batchIndex_]);  // for factored embeddings
  // clang-format on
  if(options_->hasAndNotEmpty("embedding-vectors")) {
    auto embFiles = opt<std::vector<std::string>>("embedding-vectors");
    options->set(
        "embFile", embFiles[batchIndex_], "normalization", opt<bool>("embedding-normalization"));
  }
  return New<Embedding>(graph_, options);
}

// ULR word embeddings
/*private*/ Ptr<IEmbeddingLayer> EncoderDecoderLayerBase::createULREmbeddingLayer() const {
  // clang-format off
  return New<ULREmbedding>(graph_, New<Options>(
      "dimSrcVoc",         opt<std::vector<int>>("dim-vocabs")[0],  // ULR multi-lingual src
      "dimTgtVoc",         opt<std::vector<int>>("dim-vocabs")[1],  // ULR monon tgt
      "dimUlrEmb",         opt<int>("ulr-dim-emb"),
      "dimEmb",            opt<int>("dim-emb"),
      "ulr-dropout",       opt<float>("ulr-dropout"),
      "dropout",           dropoutEmbeddings_,
      "inference",         inference_,
      "ulrTrainTransform", opt<bool>("ulr-trainable-transformation"),
      "ulrQueryFile",      opt<std::string>("ulr-query-vectors"),
      "ulrKeysFile",       opt<std::string>("ulr-keys-vectors")
    ));
  // clang-format on
}

// get embedding layer for this encoder or decoder
// This is lazy mostly because the constructors of the consuming objects are not
// guaranteed presently to have access to their graph.
Ptr<IEmbeddingLayer> EncoderDecoderLayerBase::getEmbeddingLayer(bool ulr) const {
  if(embeddingLayers_.size() <= batchIndex_ || !embeddingLayers_[batchIndex_]) {  // lazy
    if(embeddingLayers_.size() <= batchIndex_)
      embeddingLayers_.resize(batchIndex_ + 1);
    if(ulr)
      embeddingLayers_[batchIndex_] = createULREmbeddingLayer();  // embedding uses ULR
    else
      embeddingLayers_[batchIndex_] = createEmbeddingLayer();
  }
  return embeddingLayers_[batchIndex_];
}

}  // namespace marian
