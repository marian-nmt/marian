#pragma once

#include "marian.h"

#include "models/encoder.h"
#include "models/decoder.h"
#include "models/states.h"
#include "layers/constructors.h"
#include "layers/factory.h"

namespace marian {

// collection of subroutines for Transformer implementation
class Transformer {
public:
  static Expr TransposeTimeBatch(Expr input) { return transpose(input, {0, 2, 1, 3}); }

  static Expr AddPositionalEmbeddings(Ptr<ExpressionGraph> graph,
                                      Expr input,
                                      int start = 0) {
    using namespace keywords;

    int dimEmb = input->shape()[-1];
    int dimWords = input->shape()[-3];

    float num_timescales = dimEmb / 2;
    float log_timescale_increment = std::log(10000.f) / (num_timescales - 1.f);

    std::vector<float> vPos(dimEmb * dimWords, 0);
    for(int p = start; p < dimWords + start; ++p) {
      for(int i = 0; i < num_timescales; ++i) {
        float v = p * std::exp(i * -log_timescale_increment);
        vPos[(p - start) * dimEmb + i] = std::sin(v);
        vPos[(p - start) * dimEmb + num_timescales + i] = std::cos(v);
      }
    }

    // shared across batch entries
    auto signal
        = graph->constant({dimWords, 1, dimEmb}, inits::from_vector(vPos));
    return input + signal;
  }

  Expr TriangleMask(Ptr<ExpressionGraph> graph, int length) {
    using namespace keywords;

    // fill triangle mask
    std::vector<float> vMask(length * length, 0);
    for(int i = 0; i < length; ++i)
      for(int j = 0; j <= i; ++j)
        vMask[i * length + j] = 1.f;
    return graph->constant({1, length, length}, inits::from_vector(vMask));
  }

  // convert multiplicative 1/0 mask to additive 0/-inf log mask, and transpose to match result of bdot() op in Attention()
  static Expr transposedLogMask(Expr mask) { // mask: [-4: beam depth=1, -3: batch size, -2: vector dim=1, -1: max length]
    auto ms = mask->shape();
    mask = (1 - mask) * -99999999.f;
    return reshape(mask, {ms[-3], 1, ms[-2], ms[-1]}); // [-4: batch size, -3: num heads broadcast=1, -2: max length broadcast=1, -1: max length]
  }

  static Expr SplitHeads(Expr input, int dimHeads) {
    int dimModel = input->shape()[-1];
    int dimSteps = input->shape()[-2];
    int dimBatch = input->shape()[-3];
    int dimBeam = input->shape()[-4];

    int dimDepth = dimModel / dimHeads;

    auto output
        = reshape(input, {dimBatch * dimBeam, dimSteps, dimHeads, dimDepth});

    return transpose(output, {0, 2, 1, 3});
  }

  static Expr JoinHeads(Expr input, int dimBeam = 1) {
    int dimDepth = input->shape()[-1];
    int dimSteps = input->shape()[-2];
    int dimHeads = input->shape()[-3];
    int dimBatchBeam = input->shape()[-4];

    int dimModel = dimHeads * dimDepth;
    int dimBatch = dimBatchBeam / dimBeam;

    auto output = transpose(input, {0, 2, 1, 3});

    return reshape(output, {dimBeam, dimBatch, dimSteps, dimModel});
  }

  static Expr PreProcess(Ptr<ExpressionGraph> graph,
                         std::string prefix,
                         std::string ops,
                         Expr input,
                         float dropProb = 0.0f) {
    using namespace keywords;

    int dimModel = input->shape()[-1];
    auto output = input;
    for(auto op : ops) {
      // dropout
      if(op == 'd' && dropProb > 0.0f) {
        output = dropout(output, dropProb);
      }
      // layer normalization
      if(op == 'n') {
        auto scale = graph->param(
            prefix + "_ln_scale_pre", {1, dimModel}, inits::ones);
        auto bias = graph->param(
            prefix + "_ln_bias_pre", {1, dimModel}, inits::zeros);
        output = layer_norm(output, scale, bias, 1e-6);
      }
    }
    return output;
  }

  static Expr PostProcess(Ptr<ExpressionGraph> graph,
                          std::string prefix,
                          std::string ops,
                          Expr input,
                          Expr prevInput,
                          float dropProb = 0.0f) {
    using namespace keywords;

    int dimModel = input->shape()[-1];
    auto output = input;
    for(auto op : ops) {
      // dropout
      if(op == 'd' && dropProb > 0.0f) {
        output = dropout(output, dropProb);
      }
      // skip connection
      if(op == 'a') {
        output = output + prevInput;
      }
      // highway connection
      if(op == 'h') {
        auto Wh = graph->param(
            prefix + "_Wh", {dimModel, dimModel}, inits::glorot_uniform);
        auto bh = graph->param(prefix + "_bh", {1, dimModel}, inits::zeros);

        auto t = affine(prevInput, Wh, bh);
        output = highway(output, prevInput, t);
      }
      // layer normalization
      if(op == 'n') {
        auto scale
            = graph->param(prefix + "_ln_scale", {1, dimModel}, inits::ones);
        auto bias
            = graph->param(prefix + "_ln_bias", {1, dimModel}, inits::zeros);
        output = layer_norm(output, scale, bias, 1e-6);
      }
    }
    return output;
  }

  // determine the multiplicative-attention probability and performs the associative lookup as well
  // q, k, and v have already been split into multiple heads, undergone any desired linear transform.
  static Expr Attention(Ptr<ExpressionGraph> graph,
                        Ptr<Options> options,
                        std::string prefix,
                        Expr q,              // [-4: beam depth * batch size, -3: num heads, -2: max tgt length, -1: split vector dim]
                        Expr k,              // [-4: batch size, -3: num heads, -2: max src length, -1: split vector dim]
                        Expr v,              // [-4: batch size, -3: num heads, -2: max src length, -1: split vector dim]
                        Expr mask = nullptr, // [-4: batch size, -3: num heads broadcast=1, -2: max length broadcast=1, -1: max length]
                        bool inference = false) {
    using namespace keywords;

    int dk = k->shape()[-1];

    // softmax over batched dot product of query and keys (applied over all
    // time steps and batch entries), also add mask for illegal connections

    // @TODO: do this better
    int dimBeamQ = q->shape()[-4];
    int dimBeamK = k->shape()[-4];
    int dimBeam = dimBeamQ / dimBeamK;
    if(dimBeam > 1) { // broadcast k and v into all beam elements  --TODO: if we use a separate dimension, then this would be automatic at no memory cost
      k = repeat(k, dimBeam, axis = -4); // [-4: beam depth * batch size, -3: num heads, -2: max src length, -1: split vector dim]
      v = repeat(v, dimBeam, axis = -4); // [-4: beam depth * batch size, -3: num heads, -2: max src length, -1: split vector dim]
    }
    // now q, k, and v have the same first dims [-4: beam depth * batch size, -3: num heads, -2: max src or tgt length, -1: split vector dim]

    // multiplicative attention with flattened softmax
    float scale = 1.0 / std::sqrt((float)dk); // scaling to avoid extreme values due to matrix multiplication
    auto z = bdot(q, k, false, true, scale); // [-4: beam depth * batch size, -3: num heads, -2: max tgt length, -1: max src length]

    // take softmax along src sequence axis (-1)
    auto zm = z + mask;
    auto weights = softmax(zm); // [-4: beam depth * batch size, -3: num heads, -2: max tgt length, -1: max src length]

    // optional dropout for attention weights
    float dropProb
        = inference ? 0 : options->get<float>("transformer-dropout-attention");

    if(dropProb)
      weights = dropout(weights, dropProb);

    // apply attention weights to values
    return bdot(weights, v);   // [-4: beam depth * batch size, -3: num heads, -2: max tgt length, -1: split vector dim]
  }

  static Expr MultiHead(Ptr<ExpressionGraph> graph,
                        Ptr<Options> options,
                        std::string prefix,
                        int dimOut,
                        int dimHeads,
                        Expr q,                          // [-4: beam depth * batch size, -3: num heads, -2: max length, -1: split vector dim]
                        const std::vector<Expr> &keys,   // [-4: beam depth, -3: batch size, -2: max length, -1: vector dim]
                        const std::vector<Expr> &values,
                        const std::vector<Expr> &masks,  // [-4: batch size, -3: num heads broadcast=1, -2: max length broadcast=1, -1: max length]
                        bool inference = false) {
    using namespace keywords;

    int dimModel = q->shape()[-1];

    auto Wq = graph->param(
        prefix + "_Wq", {dimModel, dimModel}, inits::glorot_uniform);
    auto bq = graph->param(prefix + "_bq", {1, dimModel}, inits::zeros);
    auto qh = affine(q, Wq, bq);
    qh = SplitHeads(qh, dimHeads); // [-4: beam depth * batch size, -3: num heads, -2: max length, -1: split vector dim]

    std::vector<Expr> outputs;
    for(int i = 0; i < keys.size(); ++i) {
      std::string prefixProj = prefix;
      if(i > 0)
        prefixProj += "_enc" + std::to_string(i + 1);

      auto Wk = graph->param(prefixProj + "_Wk",
                             {dimModel, dimModel},
                             inits::glorot_uniform);
      auto bk = graph->param(
          prefixProj + "_bk", {1, dimModel}, inits::zeros);

      auto Wv = graph->param(
          prefixProj + "_Wv", {dimModel, dimModel}, inits::glorot_uniform);
      auto bv = graph->param(prefixProj + "_bv", {1, dimModel}, inits::zeros);

      auto kh = affine(keys[i], Wk, bk); // [-4: beam depth, -3: batch size, -2: max length, -1: vector dim]
      auto vh = affine(values[i], Wv, bv);

      kh = SplitHeads(kh, dimHeads); // [-4: batch size, -3: num heads, -2: max length, -1: split vector dim]
      vh = SplitHeads(vh, dimHeads); // [-4: batch size, -3: num heads, -2: max length, -1: split vector dim]

      // apply multi-head attention to downscaled inputs
      auto output
          = Attention(graph, options, prefix, qh, kh, vh, masks[i], inference); // [-4: beam depth * batch size, -3: num heads, -2: max length, -1: split vector dim]

      output = JoinHeads(output, q->shape()[-4]); // [-4: beam depth, -3: batch size, -2: max length, -1: vector dim]
      outputs.push_back(output);
    }

    Expr output;
    if(outputs.size() > 1)
      output = concatenate(outputs, axis = -1);
    else
      output = outputs.front();

    int dimAtt = output->shape()[-1];

    bool project = !options->get<bool>("transformer-no-projection");
    if(project || dimAtt != dimOut) {
      auto Wo
          = graph->param(prefix + "_Wo", {dimAtt, dimOut}, inits::glorot_uniform);
      auto bo = graph->param(prefix + "_bo", {1, dimOut}, inits::zeros);
      output = affine(output, Wo, bo);
    }

    return output;
  }

  static Expr LayerAttention(Ptr<ExpressionGraph> graph,
                             Ptr<Options> options,
                             std::string prefix,
                             Expr input,
                             Expr keys,
                             Expr values,
                             Expr mask,
                             bool inference = false) {
    return LayerAttention(graph,
                          options,
                          prefix,
                          input,
                          std::vector<Expr>{keys},
                          std::vector<Expr>{values},
                          std::vector<Expr>{mask},
                          inference);
  }

  static Expr LayerAttention(Ptr<ExpressionGraph> graph,
                             Ptr<Options> options,
                             std::string prefix,
                             Expr input,                      // [-4: beam depth, -3: batch size, -2: max length, -1: vector dim]
                             const std::vector<Expr> &keys,   // [-4: beam depth=1, -3: batch size, -2: max length, -1: vector dim]
                             const std::vector<Expr> &values,
                             const std::vector<Expr> &masks,  // [-4: batch size, -3: num heads broadcast=1, -2: max length broadcast=1, -1: max length]
                             bool inference = false) {
    using namespace keywords;

    int dimModel = input->shape()[-1];

    float dropProb = inference ? 0 : options->get<float>("transformer-dropout");
    auto opsPre = options->get<std::string>("transformer-preprocess");
    auto output = PreProcess(graph, prefix + "_Wo", opsPre, input, dropProb);

    auto heads = options->get<int>("transformer-heads");

    // multi-head self-attention over previous input
    output = MultiHead(graph,
                       options,
                       prefix,
                       dimModel,
                       heads,
                       output,
                       keys,
                       values,
                       masks,
                       inference);

    auto opsPost = options->get<std::string>("transformer-postprocess");
    output
        = PostProcess(graph, prefix + "_Wo", opsPost, output, input, dropProb);

    return output;
  }

  Expr DecoderLayerSelfAttention(rnn::State& decoderState,
                                 const rnn::State& prevDecoderState,
                                 Ptr<ExpressionGraph> graph,
                                 Ptr<Options> options,
                                 std::string prefix,
                                 Expr input,
                                 Expr selfMask,
                                 int startPos,
                                 bool inference = false) {

    using namespace keywords;

    selfMask = transposedLogMask(selfMask);

    auto values = input;
    if(startPos > 0) {
      values = concatenate({prevDecoderState.output, input},
                           axis = -2);
    }
    decoderState.output = values;

    // TODO: do not recompute matrix multiplies
    return LayerAttention(graph,
                          options,
                          prefix,
                          input,
                          values,
                          values,
                          selfMask,
                          inference);
  }

  Expr LayerFFN(Ptr<ExpressionGraph> graph,
                Ptr<Options> options,
                std::string prefix,
                Expr input,
                bool inference = false) {
    using namespace keywords;

    int dimModel = input->shape()[-1];

    float dropProb = inference ? 0 : options->get<float>("transformer-dropout");
    auto opsPre = options->get<std::string>("transformer-preprocess");
    auto output = PreProcess(graph, prefix + "_ffn", opsPre, input, dropProb);

    int dimFfn = options->get<int>("transformer-dim-ffn");
    int depthFfn = options->get<int>("transformer-ffn-depth");
    auto act = options->get<std::string>("transformer-ffn-activation");
    float ffnDropProb
      = inference ? 0 : options->get<float>("transformer-dropout-ffn");

    ABORT_IF(depthFfn < 1, "Filter depth {} is smaller than 1", depthFfn);

    int i = 1;
    int dimLast = dimModel;
    for(; i < depthFfn; ++i) {
      int dimFirst = i == 1 ? dimModel : dimFfn;
      auto W = graph->param(
          prefix + "_W" + std::to_string(i), {dimFirst, dimFfn}, inits::glorot_uniform);
      auto b = graph->param(prefix + "_b" + std::to_string(i), {1, dimFfn}, inits::zeros);

      output = affine(output, W, b);

      if(act == "relu")
        output = relu(output);
      else
        output = swish(output);

      if(ffnDropProb)
        output = dropout(output, ffnDropProb);

      dimLast = dimFfn;
    }

    auto W = graph->param(
        prefix + "_W" + std::to_string(i), {dimLast, dimModel}, inits::glorot_uniform);
    auto b = graph->param(prefix + "_b" + std::to_string(i), {1, dimModel}, inits::zeros);

    output = affine(output, W, b);

    auto opsPost = options->get<std::string>("transformer-postprocess");
    output
        = PostProcess(graph, prefix + "_ffn", opsPost, output, input, dropProb);

    return output;
  }

  Expr LayerAAN(Ptr<ExpressionGraph> graph,
                Ptr<Options> options,
                std::string prefix,
                Expr x,
                Expr y,
                bool inference = false) {
    using namespace keywords;

    int dimModel = x->shape()[-1];

    float dropProb = inference ? 0 : options->get<float>("transformer-dropout");
    auto opsPre = options->get<std::string>("transformer-preprocess");

    y = PreProcess(graph, prefix + "_ffn", opsPre, y, dropProb);

    // FFN
    int dimAan = options->get<int>("transformer-dim-aan");
    int depthAan = options->get<int>("transformer-aan-depth");
    auto act = options->get<std::string>("transformer-aan-activation");
    float aanDropProb = inference ? 0 : options->get<float>("transformer-dropout-ffn");

    int i = 1;
    int dimLast = dimModel;
    for(; i < depthAan; ++i) {
      int dimFirst = i == 1 ? dimModel : dimAan;
      auto W = graph->param(
            prefix + "_W" + std::to_string(i), {dimFirst, dimAan}, inits::glorot_uniform);
      auto b = graph->param(prefix + "_b" + std::to_string(i), {1, dimAan}, inits::zeros);

      y = affine(y, W, b);

      if(act == "relu")
        y = relu(y);
      else
        y = swish(y);

      if(aanDropProb)
        y = dropout(y, aanDropProb);

      dimLast = dimAan;
    }

    if(dimLast != dimModel) {
      auto W = graph->param(
        prefix + "_W" + std::to_string(i), {dimLast, dimModel}, inits::glorot_uniform);
      auto b = graph->param(prefix + "_b" + std::to_string(i), {1, dimModel}, inits::zeros);
      y = affine(y, W, b);
    }

    bool noGate = options->get<bool>("transformer-aan-nogate");
    if(!noGate) {
      auto Wi = graph->param(prefix + "_Wi", {dimModel, dimModel}, inits::glorot_uniform);
      auto bi = graph->param(prefix + "_bi", {1, dimModel}, inits::zeros);

      auto Wf = graph->param(prefix + "_Wf", {dimModel, dimModel}, inits::glorot_uniform);
      auto bf = graph->param(prefix + "_bf", {1, dimModel}, inits::zeros);

      auto gi = logit(affine(x, Wi, bi));
      auto gf = logit(affine(y, Wf, bf));
      y = gi * x + gf * y;
    }

    auto opsPost = options->get<std::string>("transformer-postprocess");
    y = PostProcess(graph, prefix + "_ffn", opsPost, y, x, dropProb);

    return y;
  }

  Expr DecoderLayerAAN(rnn::State& decoderState,
                       const rnn::State& prevDecoderState,
                       Ptr<ExpressionGraph> graph,
                       Ptr<Options> options,
                       std::string prefix,
                       Expr input,
                       Expr selfMask,
                       int startPos,
                       bool inference = false) {

    using namespace keywords;

    auto output = input;
    if(startPos > 0) {
      // we are decoding at a position after 0
      output = (prevDecoderState.output * startPos + input) / (startPos + 1);
    }
    else if(startPos == 0 && output->shape()[-2] > 1) {
      // we are training or scoring, because there is no history and
      // the context is larger than a single time step. We do not need
      // to average batch with only single words.
      selfMask = selfMask / sum(selfMask, axis=-1);
      output = bdot(selfMask, output);
    }
    decoderState.output = output;

    return LayerAAN(graph, options, prefix, input, output, inference);
  }

  Expr LayerOther(Ptr<ExpressionGraph> graph,
                  Ptr<Options> options,
                  std::string prefix,
                  Expr input,
                  Expr output,
                  bool inference = false) {
    using namespace keywords;

    int dimModel = input->shape()[-1];

    float dropProb = inference ? 0 : options->get<float>("transformer-dropout");
    auto opsPre = options->get<std::string>("transformer-preprocess");

    output = PreProcess(graph, prefix + "_ffn", opsPre, output, dropProb);

    auto opsPost = options->get<std::string>("transformer-postprocess");
    output
        = PostProcess(graph, prefix + "_ffn", opsPost, output, input, dropProb);

    return output;
  }
};

class EncoderTransformer : public EncoderBase, public Transformer {
public:
  EncoderTransformer(Ptr<Options> options) : EncoderBase(options) {}

  Expr WordEmbeddings(Ptr<ExpressionGraph> graph,
                      Ptr<data::CorpusBatch> batch) {
    // standard encoder word embeddings

    int dimVoc = opt<std::vector<int>>("dim-vocabs")[batchIndex_];
    int dimEmb = opt<int>("dim-emb");

    auto embFactory = embedding(graph)("dimVocab", dimVoc)("dimEmb", dimEmb);

    if(opt<bool>("tied-embeddings-src") || opt<bool>("tied-embeddings-all"))
      embFactory("prefix", "Wemb");
    else
      embFactory("prefix", prefix_ + "_Wemb");

    if(options_->has("embedding-fix-src"))
      embFactory("fixed", opt<bool>("embedding-fix-src"));

    if(options_->has("embedding-vectors")) {
      auto embFiles = opt<std::vector<std::string>>("embedding-vectors");
      embFactory                              //
          ("embFile", embFiles[batchIndex_])  //
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

    // embed the source words in the batch
    Expr batchEmbeddings, batchMask;
    std::tie(batchEmbeddings, batchMask)
        = EncoderBase::lookup(graph, embeddings, batch);

    // apply dropout over source words
    float dropoutSrc = inference_ ? 0 : opt<float>("dropout-src");
    if(dropoutSrc) {
      int srcWords = batchEmbeddings->shape()[-3];
      batchEmbeddings = dropout(batchEmbeddings, dropoutSrc, {srcWords, 1, 1});
    }

    // according to paper embeddings are scaled up by \sqrt(d_m)
    auto scaledEmbeddings = std::sqrt(dimEmb) * batchEmbeddings;

    scaledEmbeddings = AddPositionalEmbeddings(graph, scaledEmbeddings);

    // reorganize batch and timestep
    scaledEmbeddings = atleast_nd(scaledEmbeddings, 4);
    batchMask = atleast_nd(batchMask, 4);
    auto layer = TransposeTimeBatch(scaledEmbeddings); // [-4: beam depth=1, -3: batch size, -2: max length, -1: vector dim]
    auto layerMask
        = reshape(TransposeTimeBatch(batchMask), {1, dimBatch, 1, dimSrcWords}); // [-4: beam depth=1, -3: batch size, -2: vector dim=1, -1: max length]

    auto opsEmb = opt<std::string>("transformer-postprocess-emb");

    float dropProb = inference_ ? 0 : opt<float>("transformer-dropout");
    layer = PreProcess(graph, prefix_ + "_emb", opsEmb, layer, dropProb);

    layerMask = transposedLogMask(layerMask); // [-4: batch size, -3: 1, -2: vector dim=1, -1: max length]

    // apply encoder layers
    auto encDepth = opt<int>("enc-depth");
    for(int i = 1; i <= encDepth; ++i) {
      layer = LayerAttention(graph,
                             options_,
                             prefix_ + "_l" + std::to_string(i) + "_self",
                             layer,
                             layer,
                             layer,
                             layerMask,
                             inference_);

      layer = LayerFFN(graph,
                       options_,
                       prefix_ + "_l" + std::to_string(i) + "_ffn",
                       layer,
                       inference_);
    }

    // restore organization of batch and time steps. This is currently required
    // to make RNN-based decoders and beam search work with this. We are looking
    // into making this more natural.
    auto context = TransposeTimeBatch(layer); // [-4: beam depth=1, -3: max length, -2: batch size, -1: vector dim]

    return New<EncoderState>(context, batchMask, batch);
  }

  void clear() {}
};

class TransformerState : public DecoderState {
public:
  TransformerState(const rnn::States &states,
                   Expr probs,
                   std::vector<Ptr<EncoderState>> &encStates,
                   Ptr<data::CorpusBatch> batch)
      : DecoderState(states, probs, encStates, batch) {}

  virtual Ptr<DecoderState> select(const std::vector<size_t> &selIdx,
                                   int beamSize) {
    rnn::States selectedStates;

    int dimDepth = states_[0].output->shape()[-1];
    int dimTime = states_[0].output->shape()[-2];
    int dimBatch = selIdx.size() / beamSize;

    std::vector<size_t> selIdx2;
    for(auto i : selIdx)
      for(int j = 0; j < dimTime; ++j)
        selIdx2.push_back(i * dimTime + j);

    for(auto state : states_) {
      auto sel = rows(flatten_2d(state.output), selIdx2);
      sel = reshape(sel, {beamSize, dimBatch, dimTime, dimDepth});
      selectedStates.push_back({sel, nullptr});
    }

    auto selectedState = New<TransformerState>(selectedStates, probs_, encStates_, batch_);
    selectedState->setPosition(getPosition());
    return selectedState;
  }
};

class DecoderTransformer : public DecoderBase, public Transformer {
protected:
  Ptr<mlp::MLP> output_;

public:
  DecoderTransformer(Ptr<Options> options) : DecoderBase(options) {}

  virtual Ptr<DecoderState> startState(
      Ptr<ExpressionGraph> graph,
      Ptr<data::CorpusBatch> batch,
      std::vector<Ptr<EncoderState>> &encStates) {
    rnn::States startStates;
    return New<TransformerState>(startStates, nullptr, encStates, batch);
  }

  virtual Ptr<DecoderState> step(Ptr<ExpressionGraph> graph,
                                 Ptr<DecoderState> state) {
    using namespace keywords;

    auto embeddings = state->getTargetEmbeddings(); // [-4: beam depth=1, -3: max length, -2: batch size, -1: vector dim]
    auto decoderMask = state->getTargetMask();      // [max length, batch size, 1]  --this is a hypothesis

    // dropout target words
    float dropoutTrg = inference_ ? 0 : opt<float>("dropout-trg");
    if(dropoutTrg) {
      int trgWords = embeddings->shape()[-3];
      embeddings = dropout(embeddings, dropoutTrg, {trgWords, 1, 1});
    }

    //************************************************************************//

    int dimEmb = embeddings->shape()[-1];
    int dimBeam = 1;
    if(embeddings->shape().size() > 3)
      dimBeam = embeddings->shape()[-4];

    // according to paper embeddings are scaled by \sqrt(d_m)
    auto scaledEmbeddings = std::sqrt(dimEmb) * embeddings;

    int startPos = state->getPosition();

    scaledEmbeddings
        = AddPositionalEmbeddings(graph, scaledEmbeddings, startPos);

    scaledEmbeddings = atleast_nd(scaledEmbeddings, 4);

    // reorganize batch and timestep
    auto query = TransposeTimeBatch(scaledEmbeddings); // [-4: beam depth=1, -3: batch size, -2: max length, -1: vector dim]

    auto opsEmb = opt<std::string>("transformer-postprocess-emb");
    float dropProb = inference_ ? 0 : opt<float>("transformer-dropout");

    query = PreProcess(graph, prefix_ + "_emb", opsEmb, query, dropProb);

    int dimTrgWords = query->shape()[-2];
    int dimBatch = query->shape()[-3];
    auto selfMask = TriangleMask(graph, dimTrgWords);  // [ (1,) 1, max length, max length]
    if(decoderMask) {
      decoderMask = atleast_nd(decoderMask, 4);             // [ 1, max length, batch size, 1 ]
      decoderMask = reshape(TransposeTimeBatch(decoderMask),// [ 1, batch size, max length, 1 ]
                            {1, dimBatch, 1, dimTrgWords}); // [ 1, batch size, 1, max length ]
      selfMask = selfMask * decoderMask;
      // if(dimBeam > 1)
      //  selfMask = repeat(selfMask, dimBeam, axis = -4);
    }

    std::vector<Expr> encoderContexts;
    std::vector<Expr> encoderMasks;

    for(auto encoderState : state->getEncoderStates()) {
      auto encoderContext = encoderState->getContext();
      auto encoderMask = encoderState->getMask();

      encoderContext = TransposeTimeBatch(encoderContext); // [-4: beam depth=1, -3: batch size, -2: max length, -1: vector dim]

      int dimSrcWords = encoderContext->shape()[-2];

      int dims = encoderMask->shape().size();
      encoderMask = atleast_nd(encoderMask, 4);
      encoderMask = reshape(TransposeTimeBatch(encoderMask),
                            {1, dimBatch, 1, dimSrcWords});
      encoderMask = transposedLogMask(encoderMask);
      if(dimBeam > 1)
        encoderMask = repeat(encoderMask, dimBeam, axis = -4);

      encoderContexts.push_back(encoderContext);
      encoderMasks.push_back(encoderMask);
    }

    rnn::States prevDecoderStates = state->getStates();
    rnn::States decoderStates;
    // apply layers
    for(int i = 1; i <= opt<int>("dec-depth"); ++i) {
      rnn::State decoderState;
      rnn::State prevDecoderState;

      if(prevDecoderStates.size() > 0)
        prevDecoderState = prevDecoderStates[i - 1];

      std::string layerType = opt<std::string>("transformer-decoder-autoreg");
      if(layerType == "self-attention") {
        query = DecoderLayerSelfAttention(decoderState,
                                          prevDecoderState,
                                          graph,
                                          options_,
                                          prefix_ + "_l" + std::to_string(i) + "_self",
                                          query,
                                          selfMask,
                                          startPos,
                                          inference_);
      } else if(layerType == "average-attention") {
        query = DecoderLayerAAN(decoderState,
                                prevDecoderState,
                                graph,
                                options_,
                                prefix_ + "_l" + std::to_string(i) + "_aan",
                                query,
                                selfMask,
                                startPos,
                                inference_);
      } else {
        ABORT("Unknown auto-regressive layer type in transformer decoder {}", layerType);
      }

      decoderStates.push_back(decoderState);

      // Iterate over multiple encoders and simply stack the attention blocks
      if(encoderContexts.size() > 0) {
        for(int j = 0; j < encoderContexts.size(); ++j) {
          std::string prefix
              = prefix_ + "_l" + std::to_string(i) + "_context";
          if(j > 0)
            prefix += "_enc" + std::to_string(j + 1);

          query = LayerAttention(graph,
                                 options_,
                                 prefix,
                                 query,
                                 encoderContexts[j],
                                 encoderContexts[j],
                                 encoderMasks[j],
                                 inference_);
        }
      }

      query = LayerFFN(graph, // [-4: beam depth=1, -3: batch size, -2: max length, -1: vector dim]
                       options_,
                       prefix_ + "_l" + std::to_string(i) + "_ffn",
                       query,
                       inference_);
    }

    auto decoderContext = TransposeTimeBatch(query); // [-4: beam depth=1, -3: max length, -2: batch size, -1: vector dim]

    //************************************************************************//

    if(!output_) {
      int dimTrgVoc = opt<std::vector<int>>("dim-vocabs")[batchIndex_];

      auto layerOut = mlp::output(graph)         //
          ("prefix", prefix_ + "_ff_logit_out")  //
          ("dim", dimTrgVoc);

      if(opt<bool>("tied-embeddings") || opt<bool>("tied-embeddings-all")) {
        std::string tiedPrefix = prefix_ + "_Wemb";
        if(opt<bool>("tied-embeddings-all") || opt<bool>("tied-embeddings-src"))
          tiedPrefix = "Wemb";
        layerOut.tie_transposed("W", tiedPrefix);
      }

      if(shortlist_)
        layerOut.set_shortlist(shortlist_);

      // [-4: beam depth=1, -3: max length, -2: batch size, -1: vocab dim]
      // assemble layers into MLP and apply to embeddings, decoder context and
      // aligned source context
      output_ = mlp::mlp(graph)       //
                .push_back(layerOut)  //
                .construct();
    }

    Expr logits = output_->apply(decoderContext);

    // return unormalized(!) probabilities
    auto nextState = New<TransformerState>(decoderStates,
                                           logits,
                                           state->getEncoderStates(),
                                           state->getBatch());
    nextState->setPosition(state->getPosition() + 1);
    return nextState;
  }

  // helper function for guided alignment
  virtual const std::vector<Expr> getAlignments(int i = 0) { return {}; }

  void clear() {
    output_ = nullptr;
  }
};
}
