// TODO: This is really a .CPP file now. I kept the .H name to minimize confusing git, until this is code-reviewed.
// This is meant to speed-up builds, and to support Ctrl-F7 to rebuild.

#pragma once

#include "marian.h"

#include "layers/constructors.h"
#include "layers/factory.h"
#include "models/decoder.h"
#include "models/encoder.h"
#include "models/states.h"
#include "models/transformer_factory.h"
#include "rnn/constructors.h"

namespace marian {

// clang-format off

// shared base class for transformer-based EncoderTransformer and DecoderTransformer
// Both classes share a lot of code. This template adds that shared code into their
// base while still deriving from EncoderBase and DecoderBase, respectively.
template<class EncoderOrDecoderBase>
class Transformer : public EncoderOrDecoderBase {
  typedef EncoderOrDecoderBase Base;

protected:
  using Base::options_; using Base::inference_;
  std::unordered_map<std::string, Expr> cache_;

  // attention weights produced by step()
  // If enabled, it is set once per batch during training, and once per step during translation.
  // It can be accessed by getAlignments(). @TODO: move into a state or return-value object
  std::vector<Expr> alignments_; // [max tgt len or 1][beam depth, max src length, batch size, 1]

  template <typename T> T opt(const std::string& key) const { Ptr<Options> options = options_; return options->get<T>(key); }  // need to duplicate, since somehow using Base::opt is not working
  // FIXME: that separate options assignment is weird

  template <typename T> T opt(const std::string& key, const T& def) const { Ptr<Options> options = options_; if (options->has(key)) return options->get<T>(key); else return def; }

  Ptr<ExpressionGraph> graph_;

public:
  Transformer(Ptr<Options> options)
    : EncoderOrDecoderBase(options) {
  }

  static Expr transposeTimeBatch(Expr input) { return transpose(input, {0, 2, 1, 3}); }

  Expr addPositionalEmbeddings(Expr input, int start = 0) const {
    int dimEmb   = input->shape()[-1];
    int dimWords = input->shape()[-3];

    float num_timescales = (float)dimEmb / 2;
    float log_timescale_increment = std::log(10000.f) / (num_timescales - 1.f);

    std::vector<float> vPos(dimEmb * dimWords, 0);
    for(int p = start; p < dimWords + start; ++p) {
      for(int i = 0; i < num_timescales; ++i) {
        float v = p * std::exp(i * -log_timescale_increment);
        vPos[(p - start) * dimEmb + i] = std::sin(v);
        vPos[(p - start) * dimEmb + (int)num_timescales + i] = std::cos(v); // @TODO: is int vs. float correct for num_timescales?
      }
    }

    // shared across batch entries
    auto signal
        = graph_->constant({dimWords, 1, dimEmb}, inits::from_vector(vPos));
    return input + signal;
  }

  Expr triangleMask(int length) const {
    // fill triangle mask
    std::vector<float> vMask(length * length, 0);
    for(int i = 0; i < length; ++i)
      for(int j = 0; j <= i; ++j)
        vMask[i * length + j] = 1.f;
    return graph_->constant({1, length, length}, inits::from_vector(vMask));
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
    int dimBeam  = input->shape()[-4];

    int dimDepth = dimModel / dimHeads;

    auto output
        = reshape(input, {dimBatch * dimBeam, dimSteps, dimHeads, dimDepth});

    return transpose(output, {0, 2, 1, 3}); // [dimBatch*dimBeam, dimHeads, dimSteps, dimDepth]
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

  // like affine() but with built-in parameters, activation, and dropout
  static inline
  Expr dense(Expr x, std::string prefix, std::string suffix, int outDim, const std::function<Expr(Expr)>& actFn = nullptr, float dropProb = 0.0f)
  {
    auto graph = x->graph();

    auto W = graph->param(prefix + "_W" + suffix, { x->shape()[-1], outDim }, inits::glorot_uniform);
    auto b = graph->param(prefix + "_b" + suffix, { 1,              outDim }, inits::zeros);

    x = affine(x, W, b);
    if (actFn)
      x = actFn(x);
    if (dropProb)
      x = dropout(x, dropProb);
    return x;
  }

  Expr layerNorm(Expr x, std::string prefix, std::string suffix = std::string()) const {
    int dimModel = x->shape()[-1];
    auto scale = graph_->param(prefix + "_ln_scale" + suffix, { 1, dimModel }, inits::ones);
    auto bias  = graph_->param(prefix + "_ln_bias"  + suffix, { 1, dimModel }, inits::zeros);
    return marian::layerNorm(x, scale, bias, 1e-6f);
  }

  Expr preProcess(std::string prefix, std::string ops, Expr input, float dropProb = 0.0f) const {
    auto output = input;
    for(auto op : ops) {
      // dropout
      if (op == 'd')
        output = dropout(output, dropProb);
      // layer normalization
      else if (op == 'n')
        output = layerNorm(output, prefix, "_pre");
      else
        ABORT("Unknown pre-processing operation '{}'", op);
    }
    return output;
  }

  Expr postProcess(std::string prefix, std::string ops, Expr input, Expr prevInput, float dropProb = 0.0f) const {
    auto output = input;
    for(auto op : ops) {
      // dropout
      if(op == 'd')
        output = dropout(output, dropProb);
      // skip connection
      else if(op == 'a')
        output = output + prevInput;
      // highway connection
      else if(op == 'h') {
        int dimModel = input->shape()[-1];
        auto t = dense(prevInput, prefix, /*suffix=*/"h", dimModel);
        output = highway(output, prevInput, t);
      }
      // layer normalization
      else if(op == 'n')
        output = layerNorm(output, prefix);
      else
        ABORT("Unknown pre-processing operation '{}'", op);
    }
    return output;
  }

  void collectOneHead(Expr weights, int dimBeam) {
    // select first head, this is arbitrary as the choice does not really matter
    auto head0 = select(weights, -3, {0}); // @TODO: implement an index() or slice() operator and use that

    int dimBatchBeam = head0->shape()[-4];
    int srcWords = head0->shape()[-1];
    int trgWords = head0->shape()[-2];
    int dimBatch = dimBatchBeam / dimBeam;

    // reshape and transpose to match the format guided_alignment expects
    head0 = reshape(head0, {dimBeam, dimBatch, trgWords, srcWords});
    head0 = transpose(head0, {0, 3, 1, 2}); // [-4: beam depth, -3: max src length, -2: batch size, -1: max tgt length]

    // save only last alignment set. For training this will be all alignments,
    // for translation only the last one. Also split alignments by target words.
    // @TODO: make splitting obsolete
    alignments_.clear();
    for(int i = 0; i < trgWords; ++i)
      alignments_.push_back(select(head0, -1, {(size_t)i})); // [tgt index][-4: beam depth, -3: max src length, -2: batch size, -1: 1]
  }

  // determine the multiplicative-attention probability and performs the associative lookup as well
  // q, k, and v have already been split into multiple heads, undergone any desired linear transform.
  Expr Attention(std::string prefix,
                 Expr q,              // [-4: beam depth * batch size, -3: num heads, -2: max tgt length, -1: split vector dim]
                 Expr k,              // [-4: batch size, -3: num heads, -2: max src length, -1: split vector dim]
                 Expr v,              // [-4: batch size, -3: num heads, -2: max src length, -1: split vector dim]
                 Expr mask = nullptr, // [-4: batch size, -3: num heads broadcast=1, -2: max length broadcast=1, -1: max length]
                 bool saveAttentionWeights = false,
                 int dimBeam = 1) {
    int dk = k->shape()[-1];

    // softmax over batched dot product of query and keys (applied over all
    // time steps and batch entries), also add mask for illegal connections

    // multiplicative attention with flattened softmax
    float scale = 1.0f / std::sqrt((float)dk); // scaling to avoid extreme values due to matrix multiplication
    auto z = bdot(q, k, false, true, scale); // [-4: beam depth * batch size, -3: num heads, -2: max tgt length, -1: max src length]

    // mask out garbage beyond end of sequences
    z = z + mask;

    // take softmax along src sequence axis (-1)
    auto weights = softmax(z); // [-4: beam depth * batch size, -3: num heads, -2: max tgt length, -1: max src length]

    if(saveAttentionWeights)
      collectOneHead(weights, dimBeam);

    // optional dropout for attention weights
    float dropProb
        = inference_ ? 0 : opt<float>("transformer-dropout-attention");
    weights = dropout(weights, dropProb);

    // apply attention weights to values
    auto output = bdot(weights, v);   // [-4: beam depth * batch size, -3: num heads, -2: max tgt length, -1: split vector dim]
    return output;
  }

  Expr MultiHead(std::string prefix,
                 int dimOut,
                 int dimHeads,
                 Expr q,             // [-4: beam depth * batch size, -3: num heads, -2: max q length, -1: split vector dim]
                 const Expr &keys,   // [-4: beam depth, -3: batch size, -2: max kv length, -1: vector dim]
                 const Expr &values, // [-4: beam depth, -3: batch size, -2: max kv length, -1: vector dim]
                 const Expr &mask,   // [-4: batch size, -3: num heads broadcast=1, -2: max length broadcast=1, -1: max length]
                 bool cache = false,
                 bool saveAttentionWeights = false) {
    int dimModel = q->shape()[-1];
    // @TODO: good opportunity to implement auto-batching here or do something manually?
    auto Wq = graph_->param(prefix + "_Wq", {dimModel, dimModel}, inits::glorot_uniform);
    auto bq = graph_->param(prefix + "_bq", {       1, dimModel}, inits::zeros);
    auto qh = affine(q, Wq, bq);
    qh = SplitHeads(qh, dimHeads); // [-4: beam depth * batch size, -3: num heads, -2: max length, -1: split vector dim]

    Expr kh;
    // Caching transformation of the encoder that should not be created again.
    // @TODO: set this automatically by memoizing encoder context and
    // memoization propagation (short-term)
    if (!cache || (cache && cache_.count(prefix + "_keys") == 0)) {
      auto Wk = graph_->param(prefix + "_Wk", {dimModel, dimModel}, inits::glorot_uniform);
      auto bk = graph_->param(prefix + "_bk", {1,        dimModel}, inits::zeros);

      kh = affine(keys,Wk, bk);      // [-4: beam depth, -3: batch size, -2: max length, -1: vector dim]
      kh = SplitHeads(kh, dimHeads); // [-4: batch size, -3: num heads, -2: max length, -1: split vector dim]
      cache_[prefix + "_keys"] = kh;
    }
    else {
      kh = cache_[prefix + "_keys"];
    }

    Expr vh;
    if (!cache || (cache && cache_.count(prefix + "_values") == 0)) {
      auto Wv = graph_->param(prefix + "_Wv", {dimModel, dimModel}, inits::glorot_uniform);
      auto bv = graph_->param(prefix + "_bv", {1,        dimModel}, inits::zeros);

      vh = affine(values, Wv, bv); // [-4: batch size, -3: num heads, -2: max length, -1: split vector dim]
      vh = SplitHeads(vh, dimHeads);
      cache_[prefix + "_values"] = vh;
    } else {
      vh = cache_[prefix + "_values"];
    }

    int dimBeam = q->shape()[-4];

    // apply multi-head attention to downscaled inputs
    auto output
        = Attention(prefix, qh, kh, vh, mask, saveAttentionWeights, dimBeam); // [-4: beam depth * batch size, -3: num heads, -2: max length, -1: split vector dim]

    output = JoinHeads(output, dimBeam); // [-4: beam depth, -3: batch size, -2: max length, -1: vector dim]

    int dimAtt = output->shape()[-1];

    bool project = !opt<bool>("transformer-no-projection");
    if(project || dimAtt != dimOut) {
      auto Wo
        = graph_->param(prefix + "_Wo", {dimAtt, dimOut}, inits::glorot_uniform);
      auto bo = graph_->param(prefix + "_bo", {1, dimOut}, inits::zeros);
      output = affine(output, Wo, bo);
    }

    return output;
  }

  Expr LayerAttention(std::string prefix,
                      Expr input,         // [-4: beam depth, -3: batch size, -2: max length, -1: vector dim]
                      const Expr& keys,   // [-4: beam depth=1, -3: batch size, -2: max length, -1: vector dim]
                      const Expr& values, // ...?
                      const Expr& mask,   // [-4: batch size, -3: num heads broadcast=1, -2: max length broadcast=1, -1: max length]
                      bool cache = false,
                      bool saveAttentionWeights = false) {
    int dimModel = input->shape()[-1];

    float dropProb = inference_ ? 0 : opt<float>("transformer-dropout");
    auto opsPre = opt<std::string>("transformer-preprocess");
    auto output = preProcess(prefix + "_Wo", opsPre, input, dropProb);

    auto heads = opt<int>("transformer-heads");

    // multi-head self-attention over previous input
    output = MultiHead(prefix, dimModel, heads, output, keys, values, mask, cache, saveAttentionWeights);

    auto opsPost = opt<std::string>("transformer-postprocess");
    output = postProcess(prefix + "_Wo", opsPost, output, input, dropProb);

    return output;
  }

  Expr DecoderLayerSelfAttention(rnn::State& decoderLayerState,
                                 const rnn::State& prevdecoderLayerState,
                                 std::string prefix,
                                 Expr input,
                                 Expr selfMask,
                                 int startPos) {
    selfMask = transposedLogMask(selfMask);

    auto values = input;
    if(startPos > 0) {
      values = concatenate({prevdecoderLayerState.output, input}, /*axis=*/-2);
    }
    decoderLayerState.output = values;

    return LayerAttention(prefix, input, values, values, selfMask,
                          /*cache=*/false);
  }

  static inline
  std::function<Expr(Expr)> activationByName(const std::string& actName)
  {
    if (actName == "relu")
      return (ActivationFunction*)relu;
    else if (actName == "swish")
      return (ActivationFunction*)swish;
    ABORT("Invalid activation name '{}'", actName);
  }

  Expr LayerFFN(std::string prefix, Expr input) const {
    int dimModel = input->shape()[-1];

    float dropProb = inference_ ? 0 : opt<float>("transformer-dropout");
    auto opsPre = opt<std::string>("transformer-preprocess");
    auto output = preProcess(prefix + "_ffn", opsPre, input, dropProb);

    int dimFfn = opt<int>("transformer-dim-ffn");
    int depthFfn = opt<int>("transformer-ffn-depth");
    auto actFn = activationByName(opt<std::string>("transformer-ffn-activation"));
    float ffnDropProb
      = inference_ ? 0 : opt<float>("transformer-dropout-ffn");

    ABORT_IF(depthFfn < 1, "Filter depth {} is smaller than 1", depthFfn);

    // the stack of FF layers
    for(int i = 1; i < depthFfn; ++i)
      output = dense(output, prefix, /*suffix=*/std::to_string(i), dimFfn, actFn, ffnDropProb);
    output = dense(output, prefix, /*suffix=*/std::to_string(depthFfn), dimModel);

    auto opsPost = opt<std::string>("transformer-postprocess");
    output
      = postProcess(prefix + "_ffn", opsPost, output, input, dropProb);

    return output;
  }

  // Implementation of Average Attention Network Layer (AAN) from
  // https://arxiv.org/pdf/1805.00631.pdf
  Expr LayerAAN(std::string prefix, Expr x, Expr y) const {
    int dimModel = x->shape()[-1];

    float dropProb = inference_ ? 0 : opt<float>("transformer-dropout");
    auto opsPre = opt<std::string>("transformer-preprocess");

    y = preProcess(prefix + "_ffn", opsPre, y, dropProb);

    // FFN
    int dimAan   = opt<int>("transformer-dim-aan");
    int depthAan = opt<int>("transformer-aan-depth");
    auto actFn = activationByName(opt<std::string>("transformer-aan-activation"));
    float aanDropProb = inference_ ? 0 : opt<float>("transformer-dropout-ffn");

    // the stack of AAN layers
    for(int i = 1; i < depthAan; ++i)
      y = dense(y, prefix, /*suffix=*/std::to_string(i), dimAan, actFn, aanDropProb);
    if(y->shape()[-1] != dimModel) // bring it back to the desired dimension if needed
      y = dense(y, prefix, std::to_string(depthAan), dimModel);

    bool noGate = opt<bool>("transformer-aan-nogate");
    if(!noGate) {
      auto gi = dense(x, prefix, /*suffix=*/"i", dimModel, (ActivationFunction*)sigmoid);
      auto gf = dense(y, prefix, /*suffix=*/"f", dimModel, (ActivationFunction*)sigmoid);
      y = gi * x + gf * y;
    }

    auto opsPost = opt<std::string>("transformer-postprocess");
    y = postProcess(prefix + "_ffn", opsPost, y, x, dropProb);

    return y;
  }

  // Implementation of Average Attention Network Layer (AAN) from
  // https://arxiv.org/pdf/1805.00631.pdf
  // Function wrapper using decoderState as input.
  Expr DecoderLayerAAN(rnn::State& decoderState,
                       const rnn::State& prevDecoderState,
                       std::string prefix,
                       Expr input,
                       Expr selfMask,
                       int startPos) const {
    auto output = input;
    if(startPos > 0) {
      // we are decoding at a position after 0
      output = (prevDecoderState.output * (float)startPos + input) / float(startPos + 1);
    }
    else if(startPos == 0 && output->shape()[-2] > 1) {
      // we are training or scoring, because there is no history and
      // the context is larger than a single time step. We do not need
      // to average batch with only single words.
      selfMask = selfMask / sum(selfMask, /*axis=*/-1);
      output = bdot(selfMask, output);
    }
    decoderState.output = output; // BUGBUG: mutable?

    return LayerAAN(prefix, input, output);
  }

  Expr DecoderLayerRNN(rnn::State& decoderState,
                       const rnn::State& prevDecoderState,
                       std::string prefix,
                       Expr input,
                       Expr selfMask,
                       int /*startPos*/) const {
    float dropoutRnn = inference_ ? 0.f : opt<float>("dropout-rnn");

    auto rnn = rnn::rnn(graph_)                                    //
        ("type", opt<std::string>("dec-cell"))                     //
        ("prefix", prefix)                                         //
        ("dimInput", opt<int>("dim-emb"))                          //
        ("dimState", opt<int>("dim-emb"))                          //
        ("dropout", dropoutRnn)                                    //
        ("layer-normalization", opt<bool>("layer-normalization"))  //
        .push_back(rnn::cell(graph_))                              //
        .construct();

    float dropProb = inference_ ? 0 : opt<float>("transformer-dropout");
    auto opsPre = opt<std::string>("transformer-preprocess");
    auto output = preProcess(prefix, opsPre, input, dropProb);

    output = transposeTimeBatch(output);
    output = rnn->transduce(output, prevDecoderState);
    decoderState = rnn->lastCellStates()[0];
    output = transposeTimeBatch(output);

    auto opsPost = opt<std::string>("transformer-postprocess");
    output = postProcess(prefix + "_ffn", opsPost, output, input, dropProb);

    return output;
  }
};

class EncoderTransformer : public Transformer<EncoderBase> {
public:
  EncoderTransformer(Ptr<Options> options) : Transformer(options) {}

  // returns the embedding matrix based on options
  // and based on batchIndex_.
  Expr wordEmbeddings(size_t subBatchIndex) const {
    // standard encoder word embeddings

    int dimVoc = opt<std::vector<int>>("dim-vocabs")[subBatchIndex];
    int dimEmb = opt<int>("dim-emb");

    auto embFactory = embedding(graph_)("dimVocab", dimVoc)("dimEmb", dimEmb);

    if(opt<bool>("tied-embeddings-src") || opt<bool>("tied-embeddings-all"))
      embFactory("prefix", "Wemb");
    else
      embFactory("prefix", prefix_ + "_Wemb");

    if(options_->has("embedding-fix-src"))
      embFactory("fixed", opt<bool>("embedding-fix-src"));

    if(options_->has("embedding-vectors")) {
      auto embFiles = opt<std::vector<std::string>>("embedding-vectors");
      embFactory                              //
          ("embFile", embFiles[subBatchIndex])  //
          ("normalization", opt<bool>("embedding-normalization"));
    }

    return embFactory.construct();
  }

  Ptr<EncoderState> build(Ptr<ExpressionGraph> graph,
                          Ptr<data::CorpusBatch> batch) override {
    graph_ = graph;
    return apply(batch);
  }

  Ptr<EncoderState> apply(Ptr<data::CorpusBatch> batch) {
    int dimEmb = opt<int>("dim-emb");
    int dimBatch = (int)batch->size();
    int dimSrcWords = (int)(*batch)[batchIndex_]->batchWidth();

    auto embeddings = wordEmbeddings(batchIndex_); // embedding matrix, considering tying and some other options

    // embed the source words in the batch
    Expr batchEmbeddings, batchMask;
    std::tie(batchEmbeddings, batchMask)
        = EncoderBase::lookup(graph_, embeddings, batch);

    // apply dropout over source words
    float dropoutSrc = inference_ ? 0 : opt<float>("dropout-src");
    if(dropoutSrc) {
      int srcWords = batchEmbeddings->shape()[-3];
      batchEmbeddings = dropout(batchEmbeddings, dropoutSrc, {srcWords, 1, 1});
    }

    // according to paper embeddings are scaled up by \sqrt(d_m)
    auto scaledEmbeddings = std::sqrt((float)dimEmb) * batchEmbeddings;

    scaledEmbeddings = addPositionalEmbeddings(scaledEmbeddings);

    // reorganize batch and timestep
    scaledEmbeddings = atleast_nd(scaledEmbeddings, 4);
    batchMask = atleast_nd(batchMask, 4);
    auto layer = transposeTimeBatch(scaledEmbeddings); // [-4: beam depth=1, -3: batch size, -2: max length, -1: vector dim]
    auto layerMask
      = reshape(transposeTimeBatch(batchMask), {1, dimBatch, 1, dimSrcWords}); // [-4: beam depth=1, -3: batch size, -2: vector dim=1, -1: max length]

    auto opsEmb = opt<std::string>("transformer-postprocess-emb");

    float dropProb = inference_ ? 0 : opt<float>("transformer-dropout");
    layer = preProcess(prefix_ + "_emb", opsEmb, layer, dropProb);

    layerMask = transposedLogMask(layerMask); // [-4: batch size, -3: 1, -2: vector dim=1, -1: max length]

    // apply encoder layers
    auto encDepth = opt<int>("enc-depth");
    for(int i = 1; i <= encDepth; ++i) {
      layer = LayerAttention(prefix_ + "_l" + std::to_string(i) + "_self",
                             layer, // query
                             layer, // keys
                             layer, // values
                             layerMask);

      layer = LayerFFN(prefix_ + "_l" + std::to_string(i) + "_ffn", layer);
    }

    // restore organization of batch and time steps. This is currently required
    // to make RNN-based decoders and beam search work with this. We are looking
    // into making this more natural.
    auto context = transposeTimeBatch(layer); // [-4: beam depth=1, -3: max length, -2: batch size, -1: vector dim]

    return New<EncoderState>(context, batchMask, batch);
  }

  void clear() override {}
};

class TransformerState : public DecoderState {
public:
  TransformerState(const rnn::States& states,
                   Expr logProbs,
                   const std::vector<Ptr<EncoderState>>& encStates,
                   Ptr<data::CorpusBatch> batch)
      : DecoderState(states, logProbs, encStates, batch) {}

  virtual Ptr<DecoderState> select(const std::vector<size_t>& selIdx,
                                   int beamSize) const override {
    // Create hypothesis-selected state based on current state and hyp indices
    auto selectedState = New<TransformerState>(states_.select(selIdx, beamSize, /*isBatchMajor=*/true), logProbs_, encStates_, batch_);

    // Set the same target token position as the current state
    // @TODO: This is the same as in base function.
    selectedState->setPosition(getPosition());
    return selectedState;
  }
};

class DecoderTransformer : public Transformer<DecoderBase> {
private:
  Ptr<mlp::MLP> output_;

private:
  void LazyCreateOutputLayer(std::string prefix)
  {
    if(output_) // create it lazily
      return;

    int dimTrgVoc = opt<std::vector<int>>("dim-vocabs")[batchIndex_];

    auto layerOut = mlp::output(graph_)        //
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
    output_ = mlp::mlp(graph_)      //
                  .push_back(layerOut)  //
                  .construct();
  }

public:
  DecoderTransformer(Ptr<Options> options) : Transformer(options) {}

  virtual Ptr<DecoderState> startState(
      Ptr<ExpressionGraph> graph,
      Ptr<data::CorpusBatch> batch,
      std::vector<Ptr<EncoderState>>& encStates) override {
    graph_ = graph;

    std::string layerType = opt<std::string>("transformer-decoder-autoreg", "self-attention");
    if (layerType == "rnn") {
      int dimBatch = (int)batch->size();
      int dim = opt<int>("dim-emb");

      auto start = graph->constant({1, 1, dimBatch, dim}, inits::zeros);
      rnn::States startStates(opt<size_t>("dec-depth"), {start, start});

      // don't use TransformerState for RNN layers
      return New<DecoderState>(startStates, nullptr, encStates, batch);
    }
    else {
      rnn::States startStates;
      return New<TransformerState>(startStates, nullptr, encStates, batch);
    }
  }

  virtual Ptr<DecoderState> step(Ptr<ExpressionGraph> graph,
                                 Ptr<DecoderState> state) override {
    ABORT_IF(graph != graph_, "An inconsistent graph parameter was passed to step().");
    LazyCreateOutputLayer(prefix_ + "_ff_logit_out");
    return step(state);
  }

  Ptr<DecoderState> step(Ptr<DecoderState> state) {
    auto embeddings  = state->getTargetEmbeddings(); // [-4: beam depth=1, -3: max length, -2: batch size, -1: vector dim]
    auto decoderMask = state->getTargetMask();       // [max length, batch size, 1]  --this is a hypothesis

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
    auto scaledEmbeddings = std::sqrt((float)dimEmb) * embeddings;

    // set current target token position during decoding or training. At training
    // this should be 0. During translation the current length of the translation.
    // Used for position embeddings and creating new decoder states.
    int startPos = (int)state->getPosition();

    scaledEmbeddings
      = addPositionalEmbeddings(scaledEmbeddings, startPos);

    scaledEmbeddings = atleast_nd(scaledEmbeddings, 4);

    // reorganize batch and timestep
    auto query = transposeTimeBatch(scaledEmbeddings); // [-4: beam depth=1, -3: batch size, -2: max length, -1: vector dim]

    auto opsEmb = opt<std::string>("transformer-postprocess-emb");
    float dropProb = inference_ ? 0 : opt<float>("transformer-dropout");

    query = preProcess(prefix_ + "_emb", opsEmb, query, dropProb);

    int dimTrgWords = query->shape()[-2];
    int dimBatch    = query->shape()[-3];
    auto selfMask = triangleMask(dimTrgWords);  // [ (1,) 1, max length, max length]
    if(decoderMask) {
      decoderMask = atleast_nd(decoderMask, 4);             // [ 1, max length, batch size, 1 ]
      decoderMask = reshape(transposeTimeBatch(decoderMask),// [ 1, batch size, max length, 1 ]
                            {1, dimBatch, 1, dimTrgWords}); // [ 1, batch size, 1, max length ]
      selfMask = selfMask * decoderMask;
    }

    std::vector<Expr> encoderContexts;
    std::vector<Expr> encoderMasks;

    for(auto encoderState : state->getEncoderStates()) {
      auto encoderContext = encoderState->getContext();
      auto encoderMask = encoderState->getMask();

      encoderContext = transposeTimeBatch(encoderContext); // [-4: beam depth=1, -3: batch size, -2: max length, -1: vector dim]

      int dimSrcWords = encoderContext->shape()[-2];

      //int dims = encoderMask->shape().size();
      encoderMask = atleast_nd(encoderMask, 4);
      encoderMask = reshape(transposeTimeBatch(encoderMask),
                            {1, dimBatch, 1, dimSrcWords});
      encoderMask = transposedLogMask(encoderMask);
      if(dimBeam > 1)
        encoderMask = repeat(encoderMask, dimBeam, /*axis=*/ -4);

      encoderContexts.push_back(encoderContext);
      encoderMasks.push_back(encoderMask);
    }

    rnn::States prevDecoderStates = state->getStates();
    rnn::States decoderStates;
    // apply decoder layers
    auto decDepth = opt<int>("dec-depth");
    std::vector<size_t> tiedLayers = opt<std::vector<size_t>>("transformer-tied-layers",
                                                              std::vector<size_t>());
    ABORT_IF(!tiedLayers.empty() && tiedLayers.size() != decDepth,
             "Specified layer tying for {} layers, but decoder has {} layers",
             tiedLayers.size(),
             decDepth);

    for(int i = 0; i < decDepth; ++i) {
      std::string layerNo = std::to_string(i + 1);
      if (!tiedLayers.empty())
        layerNo = std::to_string(tiedLayers[i]);

      rnn::State prevDecoderState;
      if(prevDecoderStates.size() > 0)
        prevDecoderState = prevDecoderStates[i];

      // self-attention
      std::string layerType = opt<std::string>("transformer-decoder-autoreg", "self-attention");
      rnn::State decoderState;
      if(layerType == "self-attention")
        query = DecoderLayerSelfAttention(decoderState, prevDecoderState, prefix_ + "_l" + layerNo + "_self", query, selfMask, startPos);
      else if(layerType == "average-attention")
        query = DecoderLayerAAN(decoderState, prevDecoderState, prefix_ + "_l" + layerNo + "_aan", query, selfMask, startPos);
      else if(layerType == "rnn")
        query = DecoderLayerRNN(decoderState, prevDecoderState, prefix_ + "_l" + layerNo + "_rnn", query, selfMask, startPos);
      else
        ABORT("Unknown auto-regressive layer type in transformer decoder {}",
              layerType);

      // source-target attention
      // Iterate over multiple encoders and simply stack the attention blocks
      if(encoderContexts.size() > 0) {
        for(size_t j = 0; j < encoderContexts.size(); ++j) { // multiple encoders are applied one after another
          std::string prefix
            = prefix_ + "_l" + layerNo + "_context";
          if(j > 0)
            prefix += "_enc" + std::to_string(j + 1);

          // if training is performed with guided_alignment or if alignment is requested during
          // decoding or scoring return the attention weights of one head of the last layer.
          // @TODO: maybe allow to return average or max over all heads?
          bool saveAttentionWeights = false;
          if(j == 0 && (options_->has("guided-alignment") || options_->has("alignment"))) {
            size_t attLayer = decDepth - 1;
            std::string gaStr = options_->get<std::string>("transformer-guided-alignment-layer");
            if(gaStr != "last")
              attLayer = std::stoull(gaStr) - 1;

            ABORT_IF(attLayer >= decDepth,
                     "Chosen layer for guided attention ({}) larger than number of layers ({})",
                     attLayer + 1, decDepth);

            saveAttentionWeights = i == attLayer;
          }

          query = LayerAttention(prefix,
                                 query,
                                 encoderContexts[j], // keys
                                 encoderContexts[j], // values
                                 encoderMasks[j],
                                 /*cache=*/true,
                                 saveAttentionWeights);
        }
      }

      // remember decoder state
      decoderStates.push_back(decoderState);

      query = LayerFFN(prefix_ + "_l" + layerNo + "_ffn", query); // [-4: beam depth=1, -3: batch size, -2: max length, -1: vector dim]
    }

    auto decoderContext = transposeTimeBatch(query); // [-4: beam depth=1, -3: max length, -2: batch size, -1: vector dim]

    //************************************************************************//

    // final feed-forward layer (output)
    Expr logits = output_->apply(decoderContext); // [-4: beam depth=1, -3: max length, -2: batch size, -1: vocab dim]

    // return unormalized(!) probabilities
    Ptr<DecoderState> nextState;
    if (opt<std::string>("transformer-decoder-autoreg", "self-attention") == "rnn") {
      nextState = New<DecoderState>(
          decoderStates, logits, state->getEncoderStates(), state->getBatch());
    } else {
      nextState = New<TransformerState>(
          decoderStates, logits, state->getEncoderStates(), state->getBatch());
    }
    nextState->setPosition(state->getPosition() + 1);
    return nextState;
  }

  // helper function for guided alignment
  // @TODO: const vector<> seems wrong. Either make it non-const or a const& (more efficient but dangerous)
  virtual const std::vector<Expr> getAlignments(int /*i*/ = 0) override {
    return alignments_;
  }

  void clear() override {
    output_ = nullptr;
    cache_.clear();
    alignments_.clear();
  }
};

// factory functions
Ptr<EncoderBase> NewEncoderTransformer(Ptr<Options> options)
{
  return New<EncoderTransformer>(options);
}

Ptr<DecoderBase> NewDecoderTransformer(Ptr<Options> options)
{
  return New<DecoderTransformer>(options);
}

// clang-format on

}  // namespace marian
