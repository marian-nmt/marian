  #pragma once

  #include "marian.h"

  namespace marian {

  class EncoderTransformer : public EncoderBase {
  public:

    EncoderTransformer(Ptr<Options> options)
     : EncoderBase(options) {}

    Expr reverseTimeBatch(Expr input) {
      int b = input->shape()[0];
      int d = input->shape()[1];
      int t = input->shape()[2];

      auto flat = reshape(input, {b * t, d});
      std::vector<size_t> indices;
      for(int i = 0; i < b; ++i)
        for(int j = 0; j < t; ++j)
          indices.push_back(j * b + i);

      auto reversed = rows(flat, indices);

      return reshape(reversed, {t, d, b});
    }

    Expr WordEmbeddings(Ptr<ExpressionGraph> graph,
                        Ptr<data::CorpusBatch> batch) {

      // standard encoder word embeddings

      int dimVoc = opt<std::vector<int>>("dim-vocabs")[batchIndex_];
      int dimEmb = opt<int>("dim-emb");

      auto embFactory = embedding(graph)
                        ("dimVocab", dimVoc)
                        ("dimEmb", dimEmb);

      if(opt<bool>("tied-embeddings-src") || opt<bool>("tied-embeddings-all"))
        embFactory("prefix", "Wemb");
      else
        embFactory("prefix", prefix_ + "_Wemb");

      if(options_->has("embedding-fix-src"))
        embFactory("fixed", opt<bool>("embedding-fix-src"));

      if(options_->has("embedding-vectors")) {
        auto embFiles = opt<std::vector<std::string>>("embedding-vectors");
        embFactory
          ("embFile", embFiles[batchIndex_])
          ("normalization", opt<bool>("embedding-normalization"));
      }

      return embFactory.construct();
    }

    Expr PositionalEmbeddings(Ptr<ExpressionGraph> graph,
                              int dimEmb, int dimWords) {
      using namespace keywords;

      // positional embeddings as described in the paper. Maybe turn this
      // into a gpu based initializer
      std::vector<float> vPos;
      for(int p = 0; p < dimWords; ++p) {
        for(int i = 0; i < dimEmb / 2; ++i) {
          float v = p / pow(10000.f, (2.f * i) / dimEmb);
          vPos.push_back(sin(v));
          vPos.push_back(cos(v));
        }
      }

      // shared across batch entries
      return graph->constant({1, dimEmb, dimWords},
                             init=inits::from_vector(vPos));
    }

    Expr Attention(Ptr<ExpressionGraph> graph,
                   Expr q, Expr k, Expr v, Expr mask = nullptr) {
      float dk = k->shape()[1];

      // scaling to avoid extreme values due to matrix multiplication
      float scale = 1.0 / std::sqrt(dk);

      // softmax over batched dot product of query and keys (applied over all
      // time steps and batch entries)
      auto weights = softmax(dot_batch(q, k, false, true, scale), mask);

      // optional dropout for attention weights
      float dropProb = inference_ ? 0 : opt<float>("dropout-rnn");
      if(dropProb) {
        auto dropMask = graph->dropout(dropProb, weights->shape());
        weights = dropout(weights, keywords::mask = dropMask);
      }

      // apply attention weights to values
      return dot_batch(weights, v);
    }

    Expr MultiHead(Ptr<ExpressionGraph> graph,
                   std::string prefix,
                   int dimHeads,
                   Expr q, Expr k, Expr v,
                   Expr mask) {
      using namespace keywords;

      int dimModel = k->shape()[1];
      bool layerNorm = opt<bool>("layer-normalization");

      std::vector<Expr> heads;

      // loop over number of heads
      for(int i = 1; i <= dimHeads; ++i) {

        // downscaling of query, key and value, separate parameters for each head
        auto Wq = graph->param(prefix + "_Wq_h" + std::to_string(i),
                               {dimModel, dimModel / dimHeads},
                               init=inits::glorot_uniform);
        auto Wk = graph->param(prefix + "_Wk_h" + std::to_string(i),
                               {dimModel, dimModel / dimHeads},
                               init=inits::glorot_uniform);
        auto Wv = graph->param(prefix + "_Wv_h" + std::to_string(i),
                               {dimModel, dimModel / dimHeads},
                               init=inits::glorot_uniform);

        auto qh = dot(q, Wq);
        auto kh = dot(k, Wk);
        auto vh = dot(v, Wv);

        // optional layer normalization, not used here in original paper
        if(layerNorm) {
          auto gamma_q = graph->param(prefix + "_Wq_gamma_h" + std::to_string(i),
                                      {1, dimModel / dimHeads},
                                      init=inits::ones);
          auto gamma_k = graph->param(prefix + "_Wk_gamma_h" + std::to_string(i),
                                      {1, dimModel / dimHeads},
                                      init=inits::ones);
          auto gamma_v = graph->param(prefix + "_Wv_gamma_h" + std::to_string(i),
                                      {1, dimModel / dimHeads},
                                      init=inits::ones);

          qh = layer_norm(qh, gamma_q);
          kh = layer_norm(kh, gamma_k);
          vh = layer_norm(vh, gamma_v);
        }

        // apply multi-head attention to downscaled inputs
        auto head = Attention(graph, qh, kh, vh, mask);
        heads.push_back(head);
      }

      auto Wo = graph->param(prefix + "_Wo", {dimModel, dimModel},
                             init=inits::glorot_uniform);

      auto output = dot(concatenate(heads, axis=1), Wo);

      if(layerNorm) {
        auto gamma_b1 = graph->param(prefix + "_Wo_gamma", {1, dimModel},
                                     init = inits::ones);
        auto beta_b1 = graph->param(prefix + "_Wo_beta", {1, dimModel},
                                    init = inits::zeros);
        output = layer_norm(output, gamma_b1, beta_b1);
      }

      return output;

    }

    Expr Layer(Ptr<ExpressionGraph> graph,
               std::string prefix,
               int h,
               Expr input, Expr mask) {
      using namespace keywords;

      int dimModel = input->shape()[1];
      float dropProb = inference_ ? 0 : opt<float>("dropout-rnn");

      // first block: multi-head self-attention over previous input
      auto output = MultiHead(graph, prefix, h, input, input, input, mask);

      // skip connection, moved being layer normalization
      if(opt<bool>("skip"))
        output = output + input;

      // optional dropout, moved to end
      if(dropProb) {
        auto dropMask = graph->dropout(dropProb, {1, dimModel, 1});
        output = dropout(output, keywords::mask = dropMask);
      }

      auto block1 = output;

      // second block: positional feed-forward network, upscaling of
      // self-attention results.
      int dimFfn = opt<int>("transformer-dim-ffn");

      auto W1 = graph->param(prefix + "_W1", {dimModel, dimFfn},
                             init=inits::glorot_uniform);
      auto b1 = graph->param(prefix + "_b1", {1, dimFfn},
                             init=inits::zeros);

      auto W2 = graph->param(prefix + "_W2", {dimFfn, dimModel},
                             init=inits::glorot_uniform);
      auto b2 = graph->param(prefix + "_b2", {1, dimModel},
                             init=inits::zeros);

      // optional layer-normalization
      bool layerNorm = opt<bool>("layer-normalization");

      if(layerNorm) {
        auto gamma1 = graph->param(prefix + "_gamma1", {1, dimFfn},
                                   init = inits::ones);
        auto beta1 = graph->param(prefix + "_beta1", {1, dimFfn},
                                  init = inits::zeros);

        auto gamma2 = graph->param(prefix + "_gamma2", {1, dimModel},
                                   init = inits::ones);
        auto beta2 = graph->param(prefix + "_beta2", {1, dimModel},
                                  init = inits::zeros);

        output = layer_norm(relu(affine(output, W1, b1)), gamma1, beta1);
        output = layer_norm(affine(output, W2, b2), gamma2, beta2);
      }
      else {
        output = affine(relu(affine(output, W1, b1)), W2, b2);
      }

      // skip connection, moved behind layer normalization
      if(opt<bool>("skip"))
        output = output + block1;

      // optional dropout, moved to end
      if(dropProb) {
        auto dropMask = graph->dropout(dropProb, {1, dimModel, 1});
        output = dropout(output, keywords::mask = dropMask);
      }

      return output;
    }

    Ptr<EncoderState> build(Ptr<ExpressionGraph> graph,
                            Ptr<data::CorpusBatch> batch) {
      using namespace keywords;

      int dimEmb = opt<int>("dim-emb");
      int dimBatch = batch->size();
      int dimSrcWords = (*batch)[batchIndex_]->batchWidth();

      auto embeddings = WordEmbeddings(graph, batch);

      // select embeddings that occur in the batch
      Expr batchEmbeddings, batchMask;
      std::tie(batchEmbeddings, batchMask)
        = EncoderBase::lookup(embeddings, batch);

      // apply dropout over source words
      float dropProb = inference_ ? 0 : opt<float>("dropout-src");
      if(dropProb) {
        auto dropMask = graph->dropout(dropProb, {1, 1, dimSrcWords});
        batchEmbeddings = dropout(batchEmbeddings, mask = dropMask);
      }

      auto posEmbeddings = PositionalEmbeddings(graph, dimEmb, dimSrcWords);

      // according to paper embeddings are scaled by \sqrt(d_m)
      auto scaledEmbeddings = std::sqrt(dimEmb) * batchEmbeddings + posEmbeddings;

      // reorganize batch and timestep
      auto layer = reverseTimeBatch(scaledEmbeddings);
      auto layerMask = reshape(reverseTimeBatch(batchMask),
                               {1, dimSrcWords, dimBatch});

      float dropProb = inference_ ? 0 : opt<float>("dropout-rnn");
      if(dropProb) {
        auto dropMask = graph->dropout(dropProb, {1, dimEmb, 1});
        layer = dropout(layer, keywords::mask = dropMask);
      }

      // apply layers
      for(int i = 1; i <= opt<int>("enc-depth"); ++i) {
        layer = Layer(graph,
                      "encoder_l" + std::to_string(i),
                      opt<int>("transformer-heads"),
                      layer,
                      layerMask);
      }

      // restore organization of batch and time steps. This is currently required
      // to make RNN-based decoders and beam search work with this. We are looking
      // into makeing this more natural.
      auto context = reverseTimeBatch(layer);
      return New<EncoderState>(context, batchMask, batch);
    }

    void clear() { }

  };

  }
