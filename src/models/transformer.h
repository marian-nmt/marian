#pragma once

#include "marian.h"

namespace marian {

class Transformer {
public:
  Expr TransposeTimeBatch(Expr input) {
    return transpose(input, {2, 1, 0, 3});
  }

  Expr AddPositionalEmbeddings(Ptr<ExpressionGraph> graph,
                               Expr input) {
    using namespace keywords;

    int dimEmb = input->shape()[1];
    int dimWords = input->shape()[2];

    float num_timescales = dimEmb / 2;
    float log_timescale_increment = std::log(10000.f) /
      (num_timescales - 1.f);

    std::vector<float> vPos(dimEmb * dimWords, 0);
    for(int p = 0; p < dimWords; ++p) {
      for(int i = 0; i < num_timescales; ++i) {
        float v = p * std::exp(i * -log_timescale_increment);
        vPos[p * dimEmb + i] = std::sin(v);
        vPos[p * dimEmb + num_timescales + i] = std::cos(v);
      }
    }

    // shared across batch entries
    auto signal = graph->constant({1, dimEmb, dimWords},
                                  init=inits::from_vector(vPos));
    return input + signal;
  }

  Expr TriangleMask(Ptr<ExpressionGraph> graph, int length) {
    using namespace keywords;

    // fill triangle mask
    std::vector<float> vMask(length * length, 0);
    for(int i = 0; i < length; ++i)
      for(int j = 0; j <= i; ++j)
        vMask[i * length + j] = 1.f;
    return graph->constant({length, length, 1},
                            init=inits::from_vector(vMask));
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

  Expr PreProcess(Ptr<ExpressionGraph> graph,
                  std::string prefix,
                  std::string ops,
                  Expr input,
                  float dropProb=0.0f) {
    using namespace keywords;

    int dimModel = input->shape()[1];
    auto output = input;
    for(auto op : ops) {
      // dropout
      if(op == 'd' && dropProb > 0.0f) {
        auto dropMask = graph->dropout(dropProb, output->shape());
        output = dropout(output, mask = dropMask);
      }
      // layer normalization
      if(op == 'n') {
        auto scale = graph->param(prefix + "_ln_scale_pre", {1, dimModel},
                                  init = inits::ones);
        auto bias = graph->param(prefix + "_ln_bias_pre", {1, dimModel},
                                  init = inits::zeros);
        output = layer_norm(output, scale, bias);
      }
    }
    return output;
  }

  Expr PostProcess(Ptr<ExpressionGraph> graph,
                   std::string prefix,
                   std::string ops,
                   Expr input,
                   Expr prevInput,
                   float dropProb=0.0f) {
    using namespace keywords;

    int dimModel = input->shape()[1];
    auto output = input;
    for(auto op : ops) {
      // dropout
      if(op == 'd' && dropProb > 0.0f) {
        auto dropMask = graph->dropout(dropProb, output->shape());
        output = dropout(output, mask = dropMask);
      }
      // skip connection, moved behind layer normalization
      if(op == 'a') {
        output = output + prevInput;
      }
      // highway connection
      if(op == 'h') {
        auto Wh = graph->param(prefix + "_Wh", {dimModel, dimModel},
                               init = inits::glorot_uniform);
        auto bh = graph->param(prefix + "_bh", {1, dimModel},
                               init = inits::zeros);

        auto h = logit(affine(prevInput, Wh, bh));
        output = output * h + prevInput * (1 - h);
      }
      // layer normalization
      if(op == 'n') {
        auto scale = graph->param(prefix + "_ln_scale", {1, dimModel},
                                  init = inits::ones);
        auto bias = graph->param(prefix + "_ln_bias", {1, dimModel},
                                  init = inits::zeros);
        output = layer_norm(output, scale, bias, 1e-6);
      }
    }
    return output;
  }

  Expr Attention(Ptr<ExpressionGraph> graph,
                 Ptr<Options> options,
                 std::string prefix,
                 Expr q, Expr k, Expr v,
                 Expr mask=nullptr,
                 bool inference=false) {
    using namespace keywords;

    float dk = k->shape()[1];

    // scaling to avoid extreme values due to matrix multiplication
    float scale = 1.0 / std::sqrt(dk);

    // convert 0/1 mask to transformer style -inf mask
    auto ms = mask->shape();
    mask = (1 - mask) * -99999999.f;
    mask = reshape(mask, {ms[0], ms[1], 1, ms[2]});

    // softmax over batched dot product of query and keys (applied over all
    // time steps and batch entries), also add mask for illegal connections
    auto weights = softmax(bdot(q, k, false, true, scale) + mask);
    //debug(weights, prefix);

    // optional dropout for attention weights
    float dropProb = inference ? 0 : options->get<float>("transformer-dropout-attention");
    if(dropProb) {
      auto dropMask = graph->dropout(dropProb, weights->shape());
      weights = dropout(weights, mask = dropMask);
    }

    // apply attention weights to values
    return bdot(weights, v);
  }

  Expr MultiHead(Ptr<ExpressionGraph> graph,
                 Ptr<Options> options,
                 std::string prefix,
                 int dimOut,
                 int dimHeads,
                 Expr q, Expr k, Expr v,
                 Expr mask=nullptr,
                 bool inference=false) {
    using namespace keywords;

    int dimModel = q->shape()[1];

    auto Wq = graph->param(prefix + "_Wq",
                           {dimModel, dimModel},
                           init=inits::glorot_uniform);
    auto bq = graph->param(prefix + "_bq",
                           {1, dimModel},
                           init=inits::zeros);

    auto Wk = graph->param(prefix + "_Wk",
                           {dimModel, dimModel},
                           init=inits::glorot_uniform);
    auto bk = graph->param(prefix + "_bk",
                           {1, dimModel},
                           init=inits::zeros);

    auto Wv = graph->param(prefix + "_Wv",
                           {dimModel, dimModel},
                           init=inits::glorot_uniform);
    auto bv = graph->param(prefix + "_bv",
                           {1, dimModel},
                           init=inits::zeros);

    auto qh = affine(q, Wq, bq);
    auto kh = affine(k, Wk, bk);
    auto vh = affine(v, Wv, bv);

    if(true) {
      auto gammaq = graph->param(prefix + "_gammaq",
                                 {1, dimModel},
                                 init=inits::ones);
      auto betaq = graph->param(prefix + "_betaq",
                                {1, dimModel},
                                init=inits::zeros);

      auto gammak = graph->param(prefix + "_gammak",
                                 {1, dimModel},
                                 init=inits::ones);
      auto betak = graph->param(prefix + "_betak",
                                {1, dimModel},
                                init=inits::zeros);

      auto gammav = graph->param(prefix + "_gammav",
                                 {1, dimModel},
                                 init=inits::ones);
      auto betav = graph->param(prefix + "_betav",
                                {1, dimModel},
                                init=inits::zeros);

      qh = layer_norm(qh, gammaq, betaq);
      kh = layer_norm(kh, gammak, betak);
      vh = layer_norm(vh, gammav, betav);
    }

    qh = SplitHeads(qh, dimHeads);
    kh = SplitHeads(kh, dimHeads);
    vh = SplitHeads(vh, dimHeads);

    // apply multi-head attention to downscaled inputs
    auto output = Attention(graph, options, prefix, qh, kh, vh, mask, inference);
    output = JoinHeads(output);

    auto Wo = graph->param(prefix + "_Wo", {dimModel, dimOut},
                           init=inits::glorot_uniform);
    auto bo = graph->param(prefix + "_bo", {1, dimOut},
                           init=inits::zeros);
    output = affine(output, Wo, bo);

    return output;
  }

  Expr LayerAttention(Ptr<ExpressionGraph> graph,
                      Ptr<Options> options,
                      std::string prefix,
                      Expr input, Expr key, Expr value,
                      Expr mask,
                      bool inference=false) {

    using namespace keywords;

    int dimModel = input->shape()[1];

    float dropProb = inference ? 0 : options->get<float>("transformer-dropout");
    auto opsPre = options->get<std::string>("transformer-preprocess");
    auto output = PreProcess(graph, prefix + "_Wo", opsPre,
                             input,
                             dropProb);

    int heads = options->get<float>("transformer-heads");

    // multi-head self-attention over previous input
    output = MultiHead(graph, options, prefix,
                       dimModel,
                       heads, output, key, value,
                       mask,
                       inference);

    auto opsPost = options->get<std::string>("transformer-postprocess");
    output = PostProcess(graph, prefix + "_Wo", opsPost,
                         output, input,
                         dropProb);

    return output;
  }

  Expr LayerFFN(Ptr<ExpressionGraph> graph,
                Ptr<Options> options,
                std::string prefix,
                Expr input,
                bool inference=false) {

    using namespace keywords;

    int dimModel = input->shape()[1];

    float dropProb = inference ? 0 : options->get<float>("transformer-dropout");
    auto opsPre = options->get<std::string>("transformer-preprocess");
    auto output = PreProcess(graph, prefix + "_ffn", opsPre,
                             input,
                             dropProb);

    int dimFfn = options->get<int>("transformer-dim-ffn");

    auto W1 = graph->param(prefix + "_W1", {dimModel, dimFfn},
                           init=inits::glorot_uniform);
    auto b1 = graph->param(prefix + "_b1", {1, dimFfn},
                           init=inits::zeros);

    auto W2 = graph->param(prefix + "_W2", {dimFfn, dimModel},
                           init=inits::glorot_uniform);
    auto b2 = graph->param(prefix + "_b2", {1, dimModel},
                           init=inits::zeros);

    output = affine(output, W1, b1);

    if(true) {
      auto gamma1 = graph->param(prefix + "_gamma1",
                                 {1, dimFfn},
                                 init=inits::ones);
      auto beta1 = graph->param(prefix + "_beta1",
                                {1, dimFfn},
                                init=inits::zeros);

      output = layer_norm(output, gamma1, beta1);
    }

    output = relu(output);
    output = affine(output, W2, b2);

    auto opsPost = options->get<std::string>("transformer-postprocess");
    output = PostProcess(graph, prefix + "_ffn", opsPost,
                         output, input,
                         dropProb);

    return output;
  }

};

class EncoderTransformer : public EncoderBase, public Transformer {
public:

  EncoderTransformer(Ptr<Options> options)
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

    // according to paper embeddings are scaled by \sqrt(d_m)
    auto scaledEmbeddings = std::sqrt(dimEmb) * batchEmbeddings;
    scaledEmbeddings = AddPositionalEmbeddings(graph, scaledEmbeddings);

    // reorganize batch and timestep
    auto layer = TransposeTimeBatch(scaledEmbeddings);
    auto layerMask = reshape(TransposeTimeBatch(batchMask),
                             {1, dimSrcWords, dimBatch});

    float dropProb = inference_ ? 0 : opt<float>("transformer-dropout");
    layer = PostProcess(graph, prefix_ + "_emb", "dn",
                        layer, layer,
                        dropProb);

    // apply layers
    for(int i = 1; i <= opt<int>("enc-depth"); ++i) {
      layer = LayerAttention(graph, options_,
                             prefix_ + "_self_l" + std::to_string(i),
                             layer, layer, layer,
                             layerMask, inference_);

      layer = LayerFFN(graph, options_,
                       prefix_ + "_ffn_l" + std::to_string(i),
                       layer, inference_);

    }

    // restore organization of batch and time steps. This is currently required
    // to make RNN-based decoders and beam search work with this. We are looking
    // into makeing this more natural.
    auto context = TransposeTimeBatch(layer);
    return New<EncoderState>(context, batchMask, batch);
  }

  void clear() { }
};


class DecoderTransformer : public DecoderBase, public Transformer {
public:
  DecoderTransformer(Ptr<Options> options)
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
    auto decoderMask = state->getTargetMask();

    //************************************************************************//

    int dimEmb = embeddings->shape()[1];
    int dimTrgWords = embeddings->shape()[2];

    // according to paper embeddings are scaled by \sqrt(d_m)
    auto scaledEmbeddings = std::sqrt(dimEmb) * embeddings;
    scaledEmbeddings = AddPositionalEmbeddings(graph, scaledEmbeddings);

    auto encoderState = state->getEncoderStates()[0];

    auto encoderContext = encoderState->getContext();
    auto encoderMask = encoderState->getMask();

    int dimSrcWords = encoderContext->shape()[2];
    int dimBatch = encoderContext->shape()[0];

    // keep this around during steps
    encoderContext = TransposeTimeBatch(encoderContext);
    encoderMask = reshape(TransposeTimeBatch(encoderMask),
                          {1, dimSrcWords, dimBatch});

    if(decoderMask)
      decoderMask = reshape(TransposeTimeBatch(decoderMask),
                            {1, dimTrgWords, dimBatch});

    // reorganize batch and timestep
    auto layer = TransposeTimeBatch(scaledEmbeddings);

    auto selfMask = TriangleMask(graph, dimTrgWords);

    if(decoderMask)
      selfMask = selfMask * decoderMask;

    float dropProb = inference_ ? 0 : opt<float>("transformer-dropout");
    layer = PostProcess(graph, prefix_ + "_emb", "dn",
                        layer, layer,
                        dropProb);

    // apply layers
    for(int i = 1; i <= opt<int>("dec-depth"); ++i) {

      layer = LayerAttention(graph, options_,
                             prefix_ + "_self_l" + std::to_string(i),
                             layer, layer, layer,
                             selfMask, inference_);

      layer = LayerAttention(graph, options_,
                             prefix_ + "_context_l" + std::to_string(i),
                             layer, encoderContext, encoderContext,
                             encoderMask, inference_);

      layer = LayerFFN(graph, options_,
                       prefix_ + "_ffn_l" + std::to_string(i),
                       layer, inference_);

    }

    rnn::States decoderStates;
    auto decoderContext = TransposeTimeBatch(layer);

    //************************************************************************//

    int dimTrgVoc = opt<std::vector<int>>("dim-vocabs").back();

    auto layerOut = mlp::dense(graph)
                    ("prefix", prefix_ + "_ff_logit_out")
                    ("dim", dimTrgVoc);

    if(opt<bool>("tied-embeddings") || opt<bool>("tied-embeddings-all")) {
      std::string tiedPrefix = prefix_ + "_Wemb";
      if(opt<bool>("tied-embeddings-all") || opt<bool>("tied-embeddings-src"))
        tiedPrefix = "Wemb";
      layerOut.tie_transposed("W", tiedPrefix);
    }

    // assemble layers into MLP and apply to embeddings, decoder context and
    // aligned source context
    auto output = mlp::mlp(graph)
                  .push_back(layerOut);

    Expr logits = output->apply(decoderContext);

    // return unormalized(!) probabilities
    return New<DecoderState>(decoderStates, logits, state->getEncoderStates());
  }

  // helper function for guided alignment
  virtual const std::vector<Expr> getAlignments(int i = 0) {
    return {};
  }

  void clear() {
  }
};

}
