// TODO: This is really a .CPP file now. I kept the .H name to minimize confusing git, until this is code-reviewed.
// This is meant to speed-up builds, and to support Ctrl-F7 to rebuild.

#pragma once

#include "marian.h"

#include "common/hash.h"
#include "layers/constructors.h"
#include "models/decoder.h"
#include "models/encoder.h"
#include "models/states.h"
#include "models/transformer_factory.h"
#include "rnn/constructors.h"
#define _USE_MATH_DEFINES  // enables math constants. We need M_PI_2
#include <math.h>

namespace marian {

// clang-format off

// shared base class for transformer-based EncoderTransformer and DecoderTransformer
// Both classes share a lot of code. This template adds that shared code into their
// base while still deriving from EncoderBase and DecoderBase, respectively.
template<class EncoderOrDecoderBase>
class Transformer : public EncoderOrDecoderBase {
  typedef EncoderOrDecoderBase Base;
  using Base::Base;

protected:
  using Base::options_; using Base::inference_; using Base::batchIndex_; using Base::graph_;
  std::unordered_map<std::pair<std::string, Shape>, Expr> cache_;  // caching transformation of the encoder that should not be created again
  mutable/*lazy*/ std::vector<float> sinusoidalEmbeddingsFreq_, sinusoidalEmbeddingsOffs_;  // cached contributions to sinusoidal embeddings

  bool depthScaling_{false}; // As recommended in the GPT-2 paper, down-scale layer weights by a factor of 1 / sqrt(depth);
  size_t depth_{0}; // stateful depth monitoring, keep track during model construction in which layer depth we currently are. Used for depth-scaling in the formula above.

  // attention weights produced by step()
  // If enabled, it is set once per batch during training, and once per step during translation.
  // It can be accessed by getAlignments(). @TODO: move into a state or return-value object
  std::vector<Expr> alignments_; // [max tgt len or 1][beam depth, max src length, batch size, 1]

  // @TODO: make this go away
  template <typename T>
  T opt(const char* const key) const { Ptr<Options> options = options_; return options->get<T>(key); }

  template <typename T>
  T opt(const std::string& key) const { return opt<T>(key.c_str()); }

  template <typename T>
  T opt(const char* const key, const T& def) const { Ptr<Options> options = options_; return options->get<T>(key, def);  }

  template <typename T>
  T opt(const std::string& key, const T& def) const { opt<T>(key.c_str(), def); }

public:
  Transformer(Ptr<ExpressionGraph> graph, Ptr<Options> options) : EncoderOrDecoderBase(graph, options) {}

  static Expr transposeTimeBatch(Expr input) { return transpose(input, {0, 2, 1, 3}); }

  Expr addPositionalEmbeddings(Expr input, int start = 0, bool trainPosEmbeddings = false) const {
    int dimEmb   = input->shape()[-1];
    int dimWords = input->shape()[-3];

    Expr embeddings = input;

    if(trainPosEmbeddings) {
      int maxLength = opt<int>("max-length");

      // Hack for translating with length longer than trained embeddings
      // We check if the embedding matrix "Wpos" already exist so we can
      // check the number of positions in that loaded parameter.
      // We then have to restrict the maximum length to the maximum positon
      // and positions beyond this will be the maximum position.
      Expr seenEmb = graph_->get("Wpos");
      int numPos = seenEmb ? seenEmb->shape()[-2] : maxLength;

      auto embeddingLayer = embedding(
                             "prefix", "Wpos", // share positional embeddings across all encoders/decorders
                             "dimVocab", numPos,
                             "dimEmb", dimEmb)
                            .construct(graph_);

      // fill with increasing numbers until current length or maxPos
      std::vector<IndexType> positions(dimWords, numPos - 1);
      for(int i = 0; i < std::min(dimWords, numPos); ++i)
        positions[i] = i;

      auto signal = embeddingLayer->applyIndices(positions, {dimWords, 1, dimEmb});
      embeddings = embeddings + signal;
    } else {
      // @TODO : test if embeddings should be scaled when trainable
      // according to paper embeddings are scaled up by \sqrt(d_m)
      embeddings = std::sqrt((float)dimEmb) * embeddings; // embeddings were initialized to unit length; so norms will be in order of sqrt(dimEmb)

#ifdef USE_ONNX // TODO 'Sin' op and constant sine generate different result. So, use constant when 'USE_ONNX' is not defined for now.
      // precompute the arguments to sin() (the cos(x) are expressed as sin(x+pi/2))
      if (sinusoidalEmbeddingsFreq_.empty()) {
        auto numTimescales = dimEmb / 2;
        for (size_t i = 0; i < dimEmb; i++) {
          sinusoidalEmbeddingsFreq_.push_back((float)pow(1e-4, ((i % numTimescales) / (numTimescales - 1.0))));  // rotor frequency
          sinusoidalEmbeddingsOffs_.push_back((float)          ((i / numTimescales) * M_PI_2                ));  // 0 (for sin) or pi/2 (for cos)
        }
      }
      auto frequencies = graph_->constant({ dimEmb }, inits::fromVector(sinusoidalEmbeddingsFreq_));
      auto cosOffsets  = graph_->constant({ dimEmb }, inits::fromVector(sinusoidalEmbeddingsOffs_));
      auto positionRange = graph_->constant({ dimWords, 1, 1 }, inits::range((float)start, (float)start + (float)dimWords));
      positionRange->set_name("data_" + std::to_string(batchIndex_) + "_posrange");
      auto signal = sin(positionRange * frequencies + cosOffsets);
#else // USE_ONNX
      auto signal = graph_->constant({dimWords, 1, dimEmb},
                                     inits::sinusoidalPositionEmbeddings(start));
#endif // USE_ONNX

      embeddings = embeddings + signal;
    }

    return embeddings;
  }

  virtual Expr addSpecialEmbeddings(Expr input, int start = 0, Ptr<data::CorpusBatch> /*batch*/ = nullptr) const {
    bool trainPosEmbeddings = opt<bool>("transformer-train-positions", false);
    return addPositionalEmbeddings(input, start, trainPosEmbeddings);
  }

  Expr triangleMask(int length) const {
    // fill triangle mask
    std::vector<float> vMask(length * length, 0);
    for(int i = 0; i < length; ++i)
      for(int j = 0; j <= i; ++j)
        vMask[i * length + j] = 1.f;
    return graph_->constant({1, length, length}, inits::fromVector(vMask));
  }

  // convert multiplicative 1/0 mask to additive 0/-inf log mask, and transpose to match result of bdot() op in Attention()
  static Expr transposedLogMask(Expr mask) { // mask: [-4: beam depth=1, -3: batch size, -2: vector dim=1, -1: max length]
    auto ms = mask->shape();
    float maskFactor = std::max(NumericLimits<float>(mask->value_type()).lowest / 2.f, -99999999.f); // to make sure we do not overflow for fp16
    mask = (1 - mask) * maskFactor;
    return reshape(mask, {ms[-3], 1, ms[-2], ms[-1]}); // [-4: batch size, -3: num heads broadcast=1, -2: max length broadcast=1, -1: max length]
  }

  static Expr SplitHeads(Expr input, int dimHeads) {
    int dimModel = input->shape()[-1];
    int dimSteps = input->shape()[-2];
    int dimBatch = input->shape()[-3];
    int dimBeam  = input->shape()[-4];

    int dimDepth = dimModel / dimHeads;

    auto output = reshape(input, {dimBatch * dimBeam, dimSteps, dimHeads, dimDepth});

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

  Expr preProcess(std::string prefix, std::string ops, Expr input, float dropProb = 0.0f) const {
    auto output = input;
    for(auto op : ops) {
      // dropout
      if (op == 'd')
        output = dropout(output, dropProb);
      // layer normalization
      else if (op == 'n')
        output = layerNorm(output, prefix, "_pre");
      else if (op == 'r')
        output = rmsNorm(output, prefix, "_pre");
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
        auto initFn = inits::glorotUniform(true, true, depthScaling_ ? 1.f / sqrtf((float)depth_) : 1.f);
        auto t = denseInline(prevInput, prefix, /*suffix=*/"h", dimModel, initFn);
        output = highway(output, prevInput, t);
      }
      // layer normalization
      else if(op == 'n')
        output = layerNorm(output, prefix);
      else if(op == 'r')
        output = rmsNorm(output, prefix);
      else
        ABORT("Unknown pre-processing operation '{}'", op);
    }
    return output;
  }

  void collectOneHead(Expr weights, int dimBeam) {
    // select first head, this is arbitrary as the choice does not really matter
    auto head0 = slice(weights, -3, 0);

    int dimBatchBeam = head0->shape()[-4];
    int srcWords = head0->shape()[-1]; // (max) length of src sequence
    int trgWords = head0->shape()[-2]; // (max) length of trg sequence, or 1 in decoding
    int dimBatch = dimBatchBeam / dimBeam;

    // reshape and transpose to match the format guided_alignment expects
    head0 = reshape(head0, {dimBeam, dimBatch, trgWords, srcWords});
    head0 = transpose(head0, {0, 3, 1, 2}); // [beam depth, max src length, batch size, max tgt length]

    // save only last alignment set. For training this will be all alignments,
    // for translation only the last one. Also split alignments by target words.
    // @TODO: make splitting obsolete
    alignments_.clear();
    for(int i = 0; i < trgWords; ++i) { // loop over all trg positions. In decoding, there is only one.
      alignments_.push_back(slice(head0, -1, i)); // [tgt index][beam depth, max src length, batch size, 1] P(src pos|trg pos, beam index, batch index)
    }
  }

  // determine the multiplicative-attention probability and performs the associative lookup as well
  // q, k, and v have already been split into multiple heads, undergone any desired linear transform.
  Expr Attention(std::string /*prefix*/,
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
    auto z = bdot_legacy(q, k, false, true, scale); // [-4: beam depth * batch size, -3: num heads, -2: max tgt length, -1: max src length]

    // mask out garbage beyond end of sequences
    z = z + mask;

    // take softmax along src sequence axis (-1)
    auto weights = softmax(z); // [-4: beam depth * batch size, -3: num heads, -2: max tgt length, -1: max src length]

    if(saveAttentionWeights)
      collectOneHead(weights, dimBeam);

    // optional dropout for attention weights
    weights = dropout(weights, inference_ ? 0 : opt<float>("transformer-dropout-attention"));

    // apply attention weights to values
    auto output = bdot_legacy(weights, v);   // [-4: beam depth * batch size, -3: num heads, -2: max tgt length, -1: split vector dim]

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
    auto Wq = graph_->param(prefix + "_Wq", {dimModel, dimModel}, inits::glorotUniform(true, true, depthScaling_ ? 1.f / sqrtf((float)depth_) : 1.f));
    auto bq = graph_->param(prefix + "_bq", {       1, dimModel}, inits::zeros());
    auto qh = affine(q, Wq, bq);
    qh = SplitHeads(qh, dimHeads); // [-4: beam depth * batch size, -3: num heads, -2: max length, -1: split vector dim]

    Expr kh;
    // Caching transformation of the encoder that should not be created again.
    // @TODO: set this automatically by memoizing encoder context and
    // memoization propagation (short-term)
    std::pair<std::unordered_map<std::pair<std::string, Shape>, Expr>::iterator, bool> cache_result;
    if (cache
        && !((cache_result = cache_.insert(std::pair<std::pair<std::string, Shape>, Expr>({prefix + "_keys", keys->shape()}, kh))).second)
       ) {
      kh = cache_result.first->second;
    } else {
      int dimKeys =  keys->shape()[-1]; // different than dimModel when using lemma and factors combined with concatenation
      auto Wk = graph_->param(prefix + "_Wk", {dimKeys, dimModel}, inits::glorotUniform(true, true, depthScaling_ ? 1.f / sqrtf((float)depth_) : 1.f));
      auto bk = graph_->param(prefix + "_bk", {1,        dimModel}, inits::zeros());

      kh = affine(keys, Wk, bk);     // [-4: beam depth, -3: batch size, -2: max length, -1: vector dim]
      kh = SplitHeads(kh, dimHeads); // [-4: batch size, -3: num heads, -2: max length, -1: split vector dim]
      if (cache) cache_result.first->second = kh;
    }

    Expr vh;
    if (cache
        && !((cache_result = cache_.insert(std::pair<std::pair<std::string, Shape>, Expr>({prefix + "_values", values->shape()}, vh))).second)
       ) {
      vh = cache_result.first->second;
    } else {
      int dimValues = values->shape()[-1]; // different than dimModel when using lemma and factors combined with concatenation
      auto Wv = graph_->param(prefix + "_Wv", {dimValues, dimModel}, inits::glorotUniform(true, true, depthScaling_ ? 1.f / sqrtf((float)depth_) : 1.f));
      auto bv = graph_->param(prefix + "_bv", {1,        dimModel}, inits::zeros());

      vh = affine(values, Wv, bv); // [-4: batch size, -3: num heads, -2: max length, -1: split vector dim]
      vh = SplitHeads(vh, dimHeads);
      if (cache) cache_result.first->second = vh;
    }

    int dimBeam = q->shape()[-4];

    // apply multi-head attention to downscaled inputs
    auto output
        = Attention(prefix, qh, kh, vh, mask, saveAttentionWeights, dimBeam); // [-4: beam depth * batch size, -3: num heads, -2: max length, -1: split vector dim]

    output = JoinHeads(output, dimBeam); // [-4: beam depth, -3: batch size, -2: max length, -1: vector dim]

    int dimAtt = output->shape()[-1];

    bool project = !opt<bool>("transformer-no-projection");
    if(project || dimAtt != dimOut) {
      auto Wo = graph_->param(prefix + "_Wo", {dimAtt, dimOut}, inits::glorotUniform(true, true, depthScaling_ ? 1.f / sqrtf((float)depth_) : 1.f));
      auto bo = graph_->param(prefix + "_bo", {1, dimOut}, inits::zeros());
      output = affine(output, Wo, bo);
    }

    return output;
  }

  // Reduce the encoder to a single sentence vector, here we just take the contextual embedding of the first word per sentence
  // Replaces cross-attention in LASER-like models
  Expr LayerPooling(std::string prefix,
                    Expr input,            // [-4: beam depth, -3: batch size, -2: max length, -1: vector dim]
                    const Expr& values) {  // [-4: beam depth=1, -3: batch size, -2: max length (src or trg), -1: vector dim]
    int dimModel = input->shape()[-1];
    auto output = slice(values, -2, 0); // Select first word [-4: beam depth, -3: batch size, -2: 1, -1: vector dim]

    int dimPool = output->shape()[-1];
    bool project = !opt<bool>("transformer-no-projection");
    if(project || dimPool != dimModel) {
      auto Wo = graph_->param(prefix + "_Wo", {dimPool, dimModel}, inits::glorotUniform(true, true, depthScaling_ ? 1.f / sqrtf((float)depth_) : 1.f));
      auto bo = graph_->param(prefix + "_bo", {1, dimModel}, inits::zeros());
      output = affine(output, Wo, bo);  // [-4: beam depth, -3: batch size, -2: 1, -1: vector dim]
    }

    auto opsPost = opt<std::string>("transformer-postprocess");
    output = postProcess(prefix + "_Wo", opsPost, output, input, 0.f);

    return output;
  }

  Expr LayerAttention(std::string prefix,
                      Expr input,         // [-4: beam depth, -3: batch size, -2: max length, -1: vector dim]
                      Expr keys,   // [-4: beam depth=1, -3: batch size, -2: max length, -1: vector dim]
                      Expr values, // ...?
                      Expr mask,   // [-4: batch size, -3: num heads broadcast=1, -2: max length broadcast=1, -1: max length]
                      int dimHeads,
                      bool cache = false,
                      bool saveAttentionWeights = false) {
    int dimModel = input->shape()[-1];

    float dropProb = inference_ ? 0 : opt<float>("transformer-dropout");
    auto opsPre = opt<std::string>("transformer-preprocess");
    auto output = preProcess(prefix + "_Wo", opsPre, input, dropProb);

    // fixes missing norm for keys and values in self-attention with pre-norm
    if(input == keys)
       keys = output;
    if(input == values)
       values = output;

    // multi-head self-attention over previous input
    output = MultiHead(prefix, dimModel, dimHeads, output, keys, values, mask, cache, saveAttentionWeights);

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
                          opt<int>("transformer-heads"), /*cache=*/false);
  }

  Expr LayerFFN(std::string prefix, Expr input, bool isDecoder=false) const {
    int dimModel = input->shape()[-1];

    float dropProb = inference_ ? 0 : opt<float>("transformer-dropout");
    auto opsPre = opt<std::string>("transformer-preprocess");
    auto output = preProcess(prefix + "_ffn", opsPre, input, dropProb);

    auto actName = opt<std::string>("transformer-ffn-activation");

    int dimFfn = opt<int>("transformer-dim-ffn");
    int depthFfn = opt<int>("transformer-ffn-depth");
    if(isDecoder) {
      int decDimFfn = opt<int>("transformer-decoder-dim-ffn", 0);
      if(decDimFfn != 0)
        dimFfn = decDimFfn;
      
      int decDepthFfn = opt<int>("transformer-decoder-ffn-depth", 0);
      if(decDepthFfn != 0)
        depthFfn = decDepthFfn;      
    }
    
    ABORT_IF(depthFfn < 1, "Filter depth {} is smaller than 1", depthFfn);
    
    float ffnDropProb = inference_ ? 0 : opt<float>("transformer-dropout-ffn");
    auto initFn = inits::glorotUniform(true, true, depthScaling_ ? 1.f / sqrtf((float)depth_) : 1.f);

    // the stack of FF layers
    for(int i = 1; i < depthFfn; ++i)
      output = denseInline(output, prefix, /*suffix=*/std::to_string(i), dimFfn, initFn, actName, ffnDropProb);
    output = denseInline(output, prefix, /*suffix=*/std::to_string(depthFfn), dimModel, initFn);

    auto opsPost = opt<std::string>("transformer-postprocess");
    output = postProcess(prefix + "_ffn", opsPost, output, input, dropProb);

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
    auto actName = opt<std::string>("transformer-aan-activation");
    float aanDropProb = inference_ ? 0 : opt<float>("transformer-dropout-ffn");

    auto initFn = inits::glorotUniform(true, true, depthScaling_ ? 1.f / sqrtf((float)depth_) : 1.f);

    // the stack of AAN layers
    for(int i = 1; i < depthAan; ++i)
      y = denseInline(y, prefix, /*suffix=*/std::to_string(i), dimAan, initFn, actName, aanDropProb);
    if(y->shape()[-1] != dimModel) // bring it back to the desired dimension if needed
      y = denseInline(y, prefix, std::to_string(depthAan), dimModel, initFn);

    bool noGate = opt<bool>("transformer-aan-nogate");
    if(!noGate) {
      auto gi = denseInline(x, prefix, /*suffix=*/"i", dimModel, initFn, "sigmoid");
      auto gf = denseInline(y, prefix, /*suffix=*/"f", dimModel, initFn, "sigmoid");
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

  Expr DecoderLayerRNN(std::unordered_map<std::string, Ptr<rnn::RNN>>& perLayerRnn, // @TODO: rewrite this whole organically grown mess
                       rnn::State& decoderState,
                       const rnn::State& prevDecoderState,
                       std::string prefix,
                       Expr input,
                       Expr /*selfMask*/,
                       int /*startPos*/) const {
    float dropoutRnn = inference_ ? 0.f : opt<float>("dropout-rnn");

    if(!perLayerRnn[prefix]) // lazily create and cache RNNs in the decoder to avoid costly recreation @TODO: turn this into class members
      perLayerRnn[prefix] = rnn::rnn(
          "type", opt<std::string>("dec-cell"),
          "prefix", prefix,
          "dimInput", opt<int>("dim-emb"),
          "dimState", opt<int>("dim-emb"),
          "dropout", dropoutRnn,
          "layer-normalization", opt<bool>("layer-normalization"))
          .push_back(rnn::cell())
          .construct(graph_);

    auto rnn = perLayerRnn[prefix];

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
  typedef Transformer<EncoderBase> Base;
  using Base::Base;
public:
  EncoderTransformer(Ptr<ExpressionGraph> graph, Ptr<Options> options) : Transformer(graph, options) {
    depthScaling_ = options_->get<bool>("transformer-depth-scaling", false);
    depth_ = 1;
  }

  virtual ~EncoderTransformer() {}

  virtual Ptr<EncoderState> build(Ptr<ExpressionGraph> graph,
                                  Ptr<data::CorpusBatch> batch) override {
    graph_ = graph;
    return apply(batch);
  }

  Ptr<EncoderState> apply(Ptr<data::CorpusBatch> batch) {
    int dimBatch = (int)batch->size();
    int dimSrcWords = (int)(*batch)[batchIndex_]->batchWidth();
    // create the embedding matrix, considering tying and some other options
    // embed the source words in the batch
    Expr batchEmbeddings, batchMask;

    auto embeddingLayer = getEmbeddingLayer(opt<bool>("ulr", false));
    std::tie(batchEmbeddings, batchMask) = embeddingLayer->apply((*batch)[batchIndex_]);
    batchEmbeddings = addSpecialEmbeddings(batchEmbeddings, /*start=*/0, batch);

    // reorganize batch and timestep
    batchEmbeddings = atleast_nd(batchEmbeddings, 4); // [beam depth=1, max length, batch size, vector dim]
    batchMask       = atleast_nd(batchMask, 4);       // [beam depth=1, max length, batch size, vector dim=1]

    auto layer     = transposeTimeBatch(batchEmbeddings); // [beam depth=1, batch size, max length, vector dim]
    auto layerMask = transposeTimeBatch(batchMask);       // [beam depth=1, batch size, max length, vector dim=1]

    auto prevLayer = layer; // keep handle to untransformed embeddings, potentially used for a final skip connection

    auto opsEmb = opt<std::string>("transformer-postprocess-emb");
    float dropProb = inference_ ? 0 : opt<float>("transformer-dropout");
    layer = preProcess(prefix_ + "_emb", opsEmb, layer, dropProb);

    // LayerAttention expects mask in a different layout
    layerMask = reshape(layerMask, {1, dimBatch, 1, dimSrcWords}); // [1,          batch size,            1,                      max length]
    layerMask = transposedLogMask(layerMask);                      // [batch size, num heads broadcast=1, max length broadcast=1, max length]

    // apply encoder layers
    // This is the Transformer Encoder stack.
    auto encDepth = opt<int>("enc-depth");
    for(int i = 1; i <= encDepth; ++i) {
      depth_ = i;

      layer = LayerAttention(prefix_ + "_l" + std::to_string(i) + "_self",
                             layer, // query
                             layer, // keys
                             layer, // values
                             layerMask, // [batch size, num heads broadcast=1, max length broadcast=1, max length]
                             opt<int>("transformer-heads"));
      layer = LayerFFN(prefix_ + "_l" + std::to_string(i) + "_ffn", layer);
      checkpoint(layer); // sets a manually specified checkpoint if gradient checkpointing is enabled, does nothing otherwise.
    }

    // this allows to run a final layernorm operation after going through the transformer layer stack.
    // By default the operations are empty, but with prenorm (--transformer-preprocess n --transformer-postprocess da)
    // it is recommended to normalize here. Can also be used to add a skip connection from the very bottom if requested.
    auto opsTop = opt<std::string>("transformer-postprocess-top", "");
    layer = postProcess(prefix_ + "_top", opsTop, layer, prevLayer, dropProb);

    // restore organization of batch and time steps. This is currently required
    // to make RNN-based decoders and beam search work with this. We are looking
    // into making this more natural.
    auto context = transposeTimeBatch(layer); // [-4: beam depth=1, -3: max length, -2: batch size, -1: vector dim]

    return New<EncoderState>(context, batchMask, batch);
  }

  virtual void clear() override {}
};

class TransformerState : public DecoderState {
public:
  TransformerState(const rnn::States& states,
                   Logits logProbs,
                   const std::vector<Ptr<EncoderState>>& encStates,
                   Ptr<data::CorpusBatch> batch)
      : DecoderState(states, logProbs, encStates, batch) {}

  virtual Ptr<DecoderState> select(const std::vector<IndexType>& hypIndices,   // [beamIndex * activeBatchSize + batchIndex]
                                   const std::vector<IndexType>& batchIndices, // [batchIndex]
                                   int beamSize) const override {

    // @TODO: code duplication with DecoderState only because of isBatchMajor=true, should rather be a contructor argument of DecoderState?

    std::vector<Ptr<EncoderState>> newEncStates;
    for(auto& es : encStates_)
      // If the size of the batch dimension of the encoder state context changed, subselect the correct batch entries
      newEncStates.push_back(es->getContext()->shape()[-2] == batchIndices.size() ? es : es->select(batchIndices));

    // Create hypothesis-selected state based on current state and hyp indices
    auto selectedState = New<TransformerState>(states_.select(hypIndices, beamSize, /*isBatchMajor=*/true), logProbs_, newEncStates, batch_);

    // Set the same target token position as the current state
    // @TODO: This is the same as in base function.
    selectedState->setPosition(getPosition());
    return selectedState;
  }
};

class DecoderTransformer : public Transformer<DecoderBase> {
  typedef Transformer<DecoderBase> Base;
  using Base::Base;
private:
  Ptr<mlp::Output> output_;

  // This caches RNN objects to avoid reconstruction between batches or deocoding steps.
  // To be removed after refactoring of transformer.h
  std::unordered_map<std::string, Ptr<rnn::RNN>> perLayerRnn_;

private:
  // @TODO: move this out for sharing with other models
  void lazyCreateOutputLayer()
  {
    if(output_) // create it lazily
      return;

    int dimTrgVoc = opt<std::vector<int>>("dim-vocabs")[batchIndex_];

    auto outputFactory = mlp::OutputFactory(
        "prefix", prefix_ + "_ff_logit_out",
        "dim", dimTrgVoc,
        "vocab", opt<std::vector<std::string>>("vocabs")[batchIndex_], // for factored outputs
        "output-omit-bias", opt<bool>("output-omit-bias", false),
        "output-approx-knn", opt<std::vector<int>>("output-approx-knn", {}),
        "lemma-dim-emb", opt<int>("lemma-dim-emb", 0), // for factored outputs
        "lemma-dependency", opt<std::string>("lemma-dependency", ""), // for factored outputs
        "factors-combine", opt<std::string>("factors-combine", "")); // for factored outputs

    if(opt<bool>("tied-embeddings") || opt<bool>("tied-embeddings-all"))
      outputFactory.tieTransposed(opt<bool>("tied-embeddings-all") || opt<bool>("tied-embeddings-src") ? "Wemb" : prefix_ + "_Wemb");

    output_ = std::dynamic_pointer_cast<mlp::Output>(outputFactory.construct(graph_)); // (construct() returns only the underlying interface)
  }

public:
  DecoderTransformer(Ptr<ExpressionGraph> graph, Ptr<Options> options) : Transformer(graph, options) {
    depthScaling_ = options_->get<bool>("transformer-depth-scaling", false);
    depth_ = 1;
  }

  virtual Ptr<DecoderState> startState(
      Ptr<ExpressionGraph> graph,
      Ptr<data::CorpusBatch> batch,
      std::vector<Ptr<EncoderState>>& encStates) override {
    graph_ = graph;

    std::string layerType = opt<std::string>("transformer-decoder-autoreg", "self-attention");
    if (layerType == "rnn") {
      int dimBatch = (int)batch->size();
      int dim = opt<int>("dim-emb");

      auto start = graph->constant({1, 1, dimBatch, dim}, inits::zeros());
      start->set_name("decoder_start_state_" + std::to_string(batchIndex_));
      rnn::States startStates(opt<size_t>("dec-depth"), {start, start});

      // don't use TransformerState for RNN layers
      return New<DecoderState>(startStates, Logits(), encStates, batch);
    }
    else {
      rnn::States startStates;
      return New<TransformerState>(startStates, Logits(), encStates, batch);
    }
  }

  virtual Ptr<DecoderState> step(Ptr<ExpressionGraph> graph,
                                 Ptr<DecoderState> state) override {
    ABORT_IF(graph != graph_, "An inconsistent graph parameter was passed to step()");
    lazyCreateOutputLayer();
    return step(state);
  }

  Ptr<DecoderState> step(Ptr<DecoderState> state) {
    auto embeddings  = state->getTargetHistoryEmbeddings(); // [-4: beam depth=1, -3: max length, -2: batch size, -1: vector dim]
    auto decoderMask = state->getTargetMask();              // [max length, batch size, 1]  --this is a hypothesis

    //************************************************************************//

    int dimBeam = 1;
    if(embeddings->shape().size() > 3)
      dimBeam = embeddings->shape()[-4];

    // set current target token position during decoding or training. At training
    // this should be 0. During translation the current length of the translation.
    // Used for position embeddings and creating new decoder states.
    int startPos = (int)state->getPosition();

    auto scaledEmbeddings = addSpecialEmbeddings(embeddings, startPos);
    scaledEmbeddings = atleast_nd(scaledEmbeddings, 4);

    // reorganize batch and timestep
    auto query = transposeTimeBatch(scaledEmbeddings); // [-4: beam depth=1, -3: batch size, -2: max length, -1: vector dim]

    auto prevQuery = query; // keep handle to untransformed embeddings, potentially used for a final skip connection

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

    // gather encoder contexts
    std::vector<Expr> encoderContexts;
    std::vector<Expr> encoderMasks;
    for(auto encoderState : state->getEncoderStates()) {
      auto encoderContext = encoderState->getContext(); // encoder output
      auto encoderMask = encoderState->getMask(); // note: may differ from Encoder self-attention mask in that additional positions are banned for cross-attention
      encoderMask = atleast_nd(encoderMask, 4);

      encoderContext = transposeTimeBatch(encoderContext); // [beam depth=1, batch size, max length, vector dim]
      encoderMask    = transposeTimeBatch(encoderMask);    // [beam depth=1, max length, batch size, vector dim=1]

      int dimSrcWords = encoderContext->shape()[-2];

      // This would happen if something goes wrong during batch pruning.
      ABORT_IF(encoderContext->shape()[-3] != dimBatch,
               "Context and query batch dimension do not match {} != {}",
               encoderContext->shape()[-3],
               dimBatch);

      // LayerAttention expects mask in a different layout
      encoderMask = reshape(encoderMask, { 1, dimBatch, 1, dimSrcWords }); // [1,          batch size,            1,                      max length]
      encoderMask = transposedLogMask(encoderMask);                        // [batch size, num heads broadcast=1, max length broadcast=1, max length]
      if(dimBeam > 1)
        encoderMask = repeat(encoderMask, dimBeam, /*axis=*/ -4);

      encoderContexts.push_back(encoderContext);
      encoderMasks.push_back(encoderMask);

      checkpoint(encoderContext);
      checkpoint(encoderMask);
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
      depth_ = i + 1;

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
        query = DecoderLayerRNN(perLayerRnn_, decoderState, prevDecoderState, prefix_ + "_l" + layerNo + "_rnn", query, selfMask, startPos);
      else
        ABORT("Unknown auto-regressive layer type in transformer decoder {}",
              layerType);

      checkpoint(query);

      // cross-attention (source-target)
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
          if(j == 0 && (options_->get("guided-alignment", std::string("none")) != "none" || options_->hasAndNotEmpty("alignment"))) {
            size_t attLayer = decDepth - 1;
            std::string gaStr = options_->get<std::string>("transformer-guided-alignment-layer", "last");
            if(gaStr != "last")
              attLayer = std::stoull(gaStr) - 1;

            ABORT_IF(attLayer >= decDepth,
                     "Chosen layer for guided attention ({}) larger than number of layers ({})",
                     attLayer + 1, decDepth);

            saveAttentionWeights = i == attLayer;
          }

          if(options_->get<bool>("transformer-pool", false)) {
            query = LayerPooling(prefix,
                                 query,
                                 encoderContexts[j]); // values
          } else {
            query = LayerAttention(prefix,
                                   query,
                                   encoderContexts[j], // keys
                                   encoderContexts[j], // values
                                   encoderMasks[j],
                                   opt<int>("transformer-heads"),
                                   /*cache=*/true,
                                   saveAttentionWeights);
          }
        }
      }

      checkpoint(query);

      // remember decoder state
      decoderStates.push_back(decoderState);

      query = LayerFFN(prefix_ + "_l" + layerNo + "_ffn", query, /*isDecoder=*/true); // [-4: beam depth=1, -3: batch size, -2: max length, -1: vector dim]

      checkpoint(query);
    }

    // This allows to run a final layernorm operation after going through the transformer layer stack.
    // By default the operations are empty, but with prenorm (--transformer-preprocess n --transformer-postprocess da)
    // it is recommended to normalize here. Can also be used to add a skip connection from the very bottom if requested.
    auto opsTop = opt<std::string>("transformer-postprocess-top", "");
    query = postProcess(prefix_ + "_top", opsTop, query, prevQuery, dropProb);

    auto decoderContext = transposeTimeBatch(query); // [-4: beam depth=1, -3: max length, -2: batch size, -1: vector dim]

    //************************************************************************//

    // final feed-forward layer (output)
    if(shortlist_)
      output_->setShortlist(shortlist_);
    auto logits = output_->applyAsLogits(decoderContext); // [-4: beam depth=1, -3: max length, -2: batch size, -1: vocab or shortlist dim]

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
    return alignments_; // [tgt index][beam depth, max src length, batch size, 1]
  }

  void clear() override {
    if (output_)
      output_->clear();
    cache_.clear();
    alignments_.clear();
    perLayerRnn_.clear(); // this needs to be cleared between batches.
    // @TODO: figure out how to detect stale nodes i.e. nodes that are referenced,
    // but where underlying memory has been deallocated by dropping all tensors
    // from a TensorAllocator object. This can happen during ExpressionGraph::clear()
  }
};

// clang-format on

}  // namespace marian
