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

    Expr Att(Ptr<ExpressionGraph> graph,
             Expr q, Expr k, Expr v) {
      float dk = k->shape()[1];
      float scale = 1.0 / std::sqrt(dk);

      auto weights = softmax(dot_batch(q, k, false, true, scale));

      float dropProb = inference_ ? 0 : opt<float>("dropout-rnn");
      if(dropProb) {
        auto dropMask = graph->dropout(dropProb, weights->shape());
        weights = dropout(weights, keywords::mask = dropMask);
      }

      return dot_batch(weights, v);
    }

    Expr MultiHead(Ptr<ExpressionGraph> graph,
                   std::string prefix,
                   int h,
                   Expr q, Expr k, Expr v) {

      int d = k->shape()[1];

      std::vector<Expr> heads;
      for(int i = 0; i < h; ++i) {
        auto Wq = graph->param(prefix + "_Wq_h" + std::to_string(i + 1), {d, d / h},
                               keywords::init=inits::glorot_uniform);
        auto Wk = graph->param(prefix + "_Wk_h" + std::to_string(i + 1), {d, d / h},
                               keywords::init=inits::glorot_uniform);
        auto Wv = graph->param(prefix + "_Wv_h" + std::to_string(i + 1), {d, d / h},
                               keywords::init=inits::glorot_uniform);

        auto head = Att(graph, dot(q, Wq), dot(k, Wk), dot(v, Wv));
        heads.push_back(head);
      }

      auto Wo = graph->param(prefix + "_Wo", {d, d},
                             keywords::init=inits::glorot_uniform);

      return dot(concatenate(heads, keywords::axis=1), Wo);
    }

    Expr Layer(Ptr<ExpressionGraph> graph,
               std::string prefix,
               int h,
               Expr input) {

      int d = input->shape()[1];
      float dropProb = inference_ ? 0 : opt<float>("dropout-rnn");

      auto block1 = MultiHead(graph, prefix, h, input, input, input);

      if(dropProb) {
        auto dropMask = graph->dropout(dropProb, {1, d, 1});
        block1 = dropout(block1, keywords::mask = dropMask);
      }

      block1 = block1 + input;

      bool layerNorm = opt<bool>("layer-normalization");
      if(layerNorm) {
        auto gamma_b1 = graph->param(prefix + "_gamma_b1", {1, d},
                                     keywords::init = inits::from_value(1.f));
        auto beta_b1 = graph->param(prefix + "_beta_b1", {1, d},
                                    keywords::init = inits::from_value(0.f));
        block1 = layer_norm(block1, gamma_b1, beta_b1);
      }

      auto W1 = graph->param(prefix + "_W1", {d, 2048},
                             keywords::init=inits::glorot_uniform);
      auto b1 = graph->param(prefix + "_b1", {1, 2048},
                             keywords::init=inits::zeros);

      auto W2 = graph->param(prefix + "_W2", {2048, d},
                             keywords::init=inits::glorot_uniform);
      auto b2 = graph->param(prefix + "_b2", {1, d},
                             keywords::init=inits::zeros);

      auto block2 = affine(relu(affine(block1, W1, b1)), W2, b2);

      if(dropProb) {
        auto dropMask = graph->dropout(dropProb, {1, d, 1});
        block2 = dropout(block2, keywords::mask = dropMask);
      }

      block2 = block2 + block1;

      if(layerNorm) {
        auto gamma_b2 = graph->param(prefix + "_gamma_b2", {1, d},
                                     keywords::init = inits::from_value(1.f));
        auto beta_b2 = graph->param(prefix + "_beta_b2", {1, d},
                                     keywords::init = inits::from_value(0.f));

        block2 = layer_norm(block2, gamma_b2, beta_b2);
      }

      return block2;
    }

    Ptr<EncoderState> build(Ptr<ExpressionGraph> graph,
                            Ptr<data::CorpusBatch> batch) {
      using namespace keywords;

      // create source embeddings
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

      auto embeddings = embFactory.construct();

      // select embeddings that occur in the batch
      Expr batchEmbeddings, batchMask;
      std::tie(batchEmbeddings, batchMask)
        = EncoderBase::lookup(embeddings, batch);

      int dimSrcWords = batchEmbeddings->shape()[2];

      // apply dropout over source words
      float dropProb = inference_ ? 0 : opt<float>("dropout-src");
      if(dropProb) {
        auto dropMask = graph->dropout(dropProb, {1, 1, dimSrcWords});
        batchEmbeddings = dropout(batchEmbeddings, mask = dropMask);
      }

      // positional embeddings, maybe turn this into a gpu based initializer
      std::vector<float> vPos;
      for(int p = 0; p < dimSrcWords; ++p) {
        for(int i = 0; i < dimEmb; ++i) {
          float v;
          if(i % 2 == 0)
            v = sin((float)p / pow((float)10000, 2 * i / (float)dimEmb));
          else
            v = cos((float)p / pow((float)10000, 2 * i / (float)dimEmb));
          vPos.push_back(v);
        }
      }
      auto posEmbeddings = graph->constant({1, dimEmb, dimSrcWords},
                                           init=inits::from_vector(vPos));

      // reorganize batch and timestep
      auto layer = reverseTimeBatch(batchEmbeddings + posEmbeddings);

      int heads = opt<int>("transformer-heads");
      int depth = opt<int>("enc-depth");

      for(int i = 1; i <= depth; ++i)
        layer = Layer(graph, "encoder_l" + std::to_string(i), heads, layer);

      // reorganize batch and timestep
      auto context = reverseTimeBatch(layer);

      return New<EncoderState>(context, batchMask, batch);
    }

    void clear() { }

  };

  }
