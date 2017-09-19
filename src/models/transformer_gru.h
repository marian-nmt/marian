#pragma once

#include "marian.h"

namespace marian {

class TransformerGRU {
public:
  Expr transposeTimeBatch(Expr input) {
    return transpose(input, {2, 1, 0, 3});
  }

  Expr PositionalEmbeddings(Ptr<ExpressionGraph> graph,
                            int dimEmb, int first, int last) {
    using namespace keywords;

    // positional embeddings as described in the paper. Maybe turn this
    // into a gpu based initializer
    std::vector<float> vPos;
    for(int p = first; p <= last; ++p) {
      for(int i = 0; i < dimEmb / 2; ++i) {
        float v = p / pow(10000.f, (2.f * i) / dimEmb);
        vPos.push_back(sin(v));
        vPos.push_back(cos(v));
      }
    }

    // shared across batch entries
    return graph->constant({1, dimEmb, last - first + 1},
                           init=inits::from_vector(vPos));
  }

  Expr Attention(Ptr<ExpressionGraph> graph,
                 Ptr<Options> options,
                 std::string prefix,
                 Expr q, Expr k, Expr v,
                 Expr mask = nullptr) {
    float dk = k->shape()[1];

    // scaling to avoid extreme values due to matrix multiplication
    float scale = 1.0 / std::sqrt(dk);

    // softmax over batched dot product of query and keys (applied over all
    // time steps and batch entries)
    auto weights = softmax(bdot(q, k, false, true, scale), mask);

    // optional dropout for attention weights
    bool inference = options->get<bool>("inference", true);
    float dropProb = inference ? 0 : options->get<float>("dropout-rnn");
    if(dropProb) {
      auto dropMask = graph->dropout(dropProb, weights->shape());
      weights = dropout(weights, keywords::mask = dropMask);
    }

    // apply attention weights to values
    return bdot(weights, v);
  }

  Expr SplitHeads(Expr input, int dimHeads) {
    int dimSteps = input->shape()[0];
    int dimModel = input->shape()[1];
    int dimBatch = input->shape()[2];

    int dimDepth = dimModel / dimHeads;

    auto output = reshape(input, {dimHeads, dimDepth, dimSteps, dimBatch});

    return transpose(output, {2, 1, 0, 3});
  }

  Expr JoinHeads(Expr input) {
    int dimSteps = input->shape()[0];
    int dimDepth = input->shape()[1];
    int dimHeads = input->shape()[2];
    int dimBatch = input->shape()[3];

    int dimModel = dimHeads * dimDepth;

    auto output = transpose(input, {2, 1, 0, 3});

    return reshape(output, {dimSteps, dimModel, dimBatch});
  }


  Expr MultiHead(Ptr<ExpressionGraph> graph,
                 Ptr<Options> options,
                 std::string prefix,
                 int dimOut,
                 int dimHeads,
                 Expr q, Expr k, Expr v,
                 Expr mask) {
    using namespace keywords;

    int dimModel = q->shape()[1];

    auto Wq = graph->param(prefix + "_Wq",
                           {dimModel, dimModel},
                           init=inits::glorot_uniform);

    auto Wk = graph->param(prefix + "_Wk",
                           {dimModel, dimModel},
                           init=inits::glorot_uniform);

    auto Wv = graph->param(prefix + "_Wv",
                           {dimModel, dimModel},
                           init=inits::glorot_uniform);

    auto qh = dot(q, Wq);
    auto kh = dot(k, Wk);
    auto vh = dot(v, Wv);

    bool layerNorm = options->get<bool>("layer-normalization");
    if(layerNorm) {
      auto gamma_q = graph->param(prefix + "_Wq_gamma",
                                  {1, dimModel},
                                  init=inits::ones);
      auto gamma_k = graph->param(prefix + "_Wk_gamma",
                                  {1, dimModel},
                                  init=inits::ones);
      auto gamma_v = graph->param(prefix + "_Wv_gamma",
                                  {1, dimModel},
                                  init=inits::ones);

      qh = layer_norm(qh, gamma_q);
      kh = layer_norm(kh, gamma_k);
      vh = layer_norm(vh, gamma_v);
    }

    qh = SplitHeads(qh, dimHeads);
    kh = SplitHeads(kh, dimHeads);
    vh = SplitHeads(vh, dimHeads);

    // apply multi-head attention to downscaled inputs
    auto output = Attention(graph, options, prefix, qh, kh, vh, mask);
    output = JoinHeads(output);

    if(dimModel != dimOut) {
      auto Wo = graph->param(prefix + "_Wo", {dimModel, dimOut},
                             init=inits::glorot_uniform);
      output = dot(output, Wo);
    }

    return output;
  }

  Expr LayerAttention(Ptr<ExpressionGraph> graph,
                      Ptr<Options> options,
                      std::string prefix,
                      int dimOut,
                      Expr input, Expr key, Expr value,
                      Expr mask,
                      bool inference=false) {

    using namespace keywords;

    int heads = options->get<float>("transformer-heads");

    // multi-head self-attention over previous input
    Expr output = MultiHead(graph, options, prefix, dimOut,
                            heads, input, key, value,
                            mask);

    bool layerNorm = options->get<bool>("layer-normalization");
    if(layerNorm) {
      auto gamma = graph->param(prefix + "_Wo_gamma", {1, dimOut},
                                init = inits::ones);
      auto beta = graph->param(prefix + "_Wo_beta", {1, dimOut},
                               init = inits::zeros);
      output = layer_norm(output, gamma, beta);
    }

    // skip connection, moved being layer normalization
    //if(options->get<bool>("skip"))
    //  output = output + input;

    // optional dropout, moved to end
    float dropProb = inference ? 0 : options->get<float>("dropout-rnn");
    if(dropProb) {
      auto dropMask = graph->dropout(dropProb, {1, dimOut, 1});
      output = dropout(output, keywords::mask = dropMask);
    }

    return output;
  }

  Expr LayerFFN(Ptr<ExpressionGraph> graph,
                Ptr<Options> options,
                std::string prefix,
                Expr input,
                bool inference=false) {

    using namespace keywords;

    int dimModel = options->get<int>("dim-emb");
    int dimFfn = options->get<int>("transformer-dim-ffn");

    auto W1 = graph->param(prefix + "_W1", {dimModel, dimFfn},
                           init=inits::glorot_uniform);
    auto b1 = graph->param(prefix + "_b1", {1, dimFfn},
                           init=inits::zeros);

    auto W2 = graph->param(prefix + "_W2", {dimFfn, dimModel},
                           init=inits::glorot_uniform);
    auto b2 = graph->param(prefix + "_b2", {1, dimModel},
                           init=inits::zeros);

    // optional layer-normalization
    bool layerNorm = options->get<bool>("layer-normalization");

    Expr output;
    if(layerNorm) {
      auto gamma1 = graph->param(prefix + "_gamma1", {1, dimFfn},
                                 init = inits::ones);
      auto beta1 = graph->param(prefix + "_beta1", {1, dimFfn},
                                init = inits::zeros);

      auto gamma2 = graph->param(prefix + "_gamma2", {1, dimModel},
                                 init = inits::ones);
      auto beta2 = graph->param(prefix + "_beta2", {1, dimModel},
                                init = inits::zeros);

      output = layer_norm(relu(affine(input, W1, b1)), gamma1, beta1);
      output = layer_norm(affine(output, W2, b2), gamma2, beta2);
    }
    else {
      output = affine(relu(affine(input, W1, b1)), W2, b2);
    }

    // skip connection, moved behind layer normalization
    if(options->get<bool>("skip"))
      output = output + input;

    // optional dropout, moved to end
    float dropProb = inference ? 0 : options->get<float>("dropout-rnn");
    if(dropProb) {
      auto dropMask = graph->dropout(dropProb, {1, dimModel, 1});
      output = dropout(output, keywords::mask = dropMask);
    }

    return output;
  }

};

class EncoderTransformerGRU : public EncoderBase, public TransformerGRU {
public:

  EncoderTransformerGRU(Ptr<Options> options)
   : EncoderBase(options) {}

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
    float dropProbSrc = inference_ ? 0 : opt<float>("dropout-src");
    if(dropProbSrc) {
      auto dropMask = graph->dropout(dropProbSrc, {1, 1, dimSrcWords});
      batchEmbeddings = dropout(batchEmbeddings, mask = dropMask);
    }

    auto posEmbeddings = PositionalEmbeddings(graph, dimEmb, 1, dimSrcWords);

    // according to paper embeddings are scaled by \sqrt(d_m)
    auto scaledEmbeddings = std::sqrt(dimEmb) * batchEmbeddings + posEmbeddings;

    // reorganize batch and timestep
    auto layer = transposeTimeBatch(scaledEmbeddings);
    auto layerMask = reshape(transposeTimeBatch(batchMask),
                             {1, dimSrcWords, dimBatch});

    float dropProb = inference_ ? 0 : opt<float>("dropout-rnn");
    if(dropProb) {
      auto dropMask = graph->dropout(dropProb, {1, dimEmb, 1});
      layer = dropout(layer, keywords::mask = dropMask);
    }

    auto fake = graph->constant({dimEmb, 3 * dimEmb}, init=inits::zeros);

    // apply layers
    for(int i = 1; i <= opt<int>("enc-depth"); ++i) {
      auto self = LayerAttention(graph, options_,
                                 prefix_ + "_self_l" + std::to_string(i),
                                 3 * dimEmb,
                                 layer, layer, layer,
                                 layerMask);

      auto bself = graph->param(prefix_ + "_selfB_l" + std::to_string(i),
                                {1, 3 * dimEmb},
                                init=inits::zeros);

      layer = rnn::gruOps({layer, fake, self, bself});

      layer = LayerFFN(graph, options_,
                       prefix_ + "_ffn_l" + std::to_string(i),
                       layer);

    }

    // restore organization of batch and time steps. This is currently required
    // to make RNN-based decoders and beam search work with this. We are looking
    // into makeing this more natural.
    auto context = transposeTimeBatch(layer);
    return New<EncoderState>(context, batchMask, batch);
  }

  void clear() { }
};


class DecoderTransformerGRU : public DecoderBase, public TransformerGRU {
public:
  DecoderTransformerGRU(Ptr<Options> options)
   : DecoderBase(options) {}

  virtual Ptr<DecoderState> startState(Ptr<ExpressionGraph> graph,
                                       Ptr<data::CorpusBatch> batch,
                                       std::vector<Ptr<EncoderState>>& encStates) {
    using namespace keywords;

    rnn::States startStates(opt<size_t>("dec-depth"), {nullptr, nullptr});
    return New<DecoderState>(startStates, nullptr, encStates);
  }

  virtual Ptr<DecoderState> step(Ptr<ExpressionGraph> graph,
                                 Ptr<DecoderState> state) {
    using namespace keywords;

    auto embeddings = state->getTargetEmbeddings();

    //************************************************************************//

    int dimEmb = embeddings->shape()[1];
    int dimTrgWords = embeddings->shape()[2];

    // generate positional embeddings and shift by one time step
    auto posEmbeddings = PositionalEmbeddings(graph, dimEmb, 0, dimTrgWords - 1);

    // according to paper embeddings are scaled by \sqrt(d_m)
    auto scaledEmbeddings = std::sqrt(dimEmb) * embeddings + posEmbeddings;

    auto encoderState = state->getEncoderStates()[0];

    auto encoderContext = encoderState->getContext();
    auto encoderMask = encoderState->getMask();

    int dimSrcWords = encoderContext->shape()[2];
    int dimBatch = encoderContext->shape()[0];

    // keep this around during steps
    encoderContext = transposeTimeBatch(encoderContext);
    encoderMask = reshape(transposeTimeBatch(encoderMask),
                          {1, dimSrcWords, dimBatch});

    // reorganize batch and timestep
    auto layer = transposeTimeBatch(scaledEmbeddings);

    // fill triangle mask
    std::vector<float> vSelfMask(dimTrgWords * dimTrgWords, 0);
    for(int i = 0; i < dimTrgWords; ++i)
      for(int j = 0; j <= i; ++j)
        vSelfMask[i * dimTrgWords + j] = 1.f;
    auto selfMask = graph->constant({dimTrgWords, dimTrgWords, 1},
                                    init=inits::from_vector(vSelfMask));

    float dropProb = inference_ ? 0 : opt<float>("dropout-rnn");
    if(dropProb) {
      auto dropMask = graph->dropout(dropProb, {1, dimEmb, 1});
      layer = dropout(layer, keywords::mask = dropMask);
    }

    // apply layers
    for(int i = 1; i <= opt<int>("dec-depth"); ++i) {

      auto self = LayerAttention(graph, options_,
                                 prefix_ + "_self_l" + std::to_string(i),
                                 3 * dimEmb,
                                 layer, layer, layer,
                                 selfMask);

      auto ctx = LayerAttention(graph, options_,
                                prefix_ + "_context_l" + std::to_string(i),
                                3 * dimEmb,
                                layer, encoderContext, encoderContext,
                                encoderMask);

      auto b = graph->param(prefix_ + "_ctxB_l" + std::to_string(i),
                            {1, 3 * dimEmb},
                            init=inits::zeros);

      layer = rnn::gruOps({layer, self, ctx, b});

      layer = LayerFFN(graph, options_,
                       prefix_ + "_ffn_l" + std::to_string(i),
                       layer);

    }

    rnn::States decoderStates;
    auto decoderContext = transposeTimeBatch(layer);

    //************************************************************************//

    // construct deep output multi-layer network layer-wise
    auto layer1 = mlp::dense(graph)
                  ("prefix", prefix_ + "_ff_logit_l1")
                  ("dim", opt<int>("dim-emb"))
                  ("activation", mlp::act::tanh)
                  ("layer-normalization", opt<bool>("layer-normalization"));

    int dimTrgVoc = opt<std::vector<int>>("dim-vocabs").back();

    auto layer2 = mlp::dense(graph)
                  ("prefix", prefix_ + "_ff_logit_l2")
                  ("dim", dimTrgVoc);

    if(opt<bool>("tied-embeddings") || opt<bool>("tied-embeddings-all")) {
      std::string tiedPrefix = prefix_ + "_Wemb";
      if(opt<bool>("tied-embeddings-all") || opt<bool>("tied-embeddings-src"))
        tiedPrefix = "Wemb";
      layer2.tie_transposed("W", tiedPrefix);
    }

    // assemble layers into MLP and apply to embeddings, decoder context and
    // aligned source context
    auto output = mlp::mlp(graph)
                  .push_back(layer1)
                  .push_back(layer2);

    Expr logits = output->apply(embeddings, decoderContext);

    // return unormalized(!) probabilities
    return New<DecoderState>(decoderStates, logits, state->getEncoderStates());
  }

  // helper function for guided alignment
  virtual const std::vector<Expr> getAlignments(int i = 0) {
    return {};
  }

  void clear() {}
};

}
