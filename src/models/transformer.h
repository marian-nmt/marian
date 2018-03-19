#pragma once

#include "marian.h"

#include "encdec.h"
#include "layers/constructors.h"
#include "layers/factory.h"
#include "model_base.h"
#include "model_factory.h"

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
                        bool inference = false, Expr* pExtraLoss = nullptr, Expr* pSentEndProb = nullptr,
                        bool isSelf = false) {
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

    // [Shaw et al.: Self-Attention with Relative Position Representations, https://arxiv.org/pdf/1803.02155.pdf]
    // relative position info comes in here
    //  - add offset embedding to v (Eq. (3))
    //  - add offset embedding to k (Eq. (4))
    // offset embedding :=
    //  - learned weights
    //  - embeddingVectors[j-i]
    //  - depends on index in both q and k/v, so must operate on dims of z, can't just add to k/v
    //  - same dimension as v.shape[-1] and likewise k
    //  - shared across the 8 heads
    //  - j-i may be clipped, e.g. to max. +/-2 (N=5 vectors only)
    //  - would a custom kernel be the easiest solution? Or a matrix product?
    //  - src = tgt
    //  - j = s, i = t
    //  - t = sequence index into q       --note: confusing naming, since src=tgt (self-attention)
    //  - s = sequence index into k and v
    //  - N := max offset embeddings (e.g. 5)
    // dimensions for z:
    //  - z[b,h,t,s] += zk[b,h,t,s]
    //  - zk[b,h,t,s] = f(q[b,h,t,.], a[s,t,.]) = q[b,h,t,.] a[s,t,.]'    --note: also the scale, left out for clarity
    //  - qw[b,h,t,N] = q[b,h,t,.] @ w[N,.]'                              --every q scored against every offset embedding
    //  - now map this (select/shuffle). Can that be a sparse matrix product?
    //  - zk[b,h,t,s] = qw[b,h,t,(s-t)]
    auto offsetEmbeddingRange = isSelf ? options->get<int>("transformer-offset-embedding-range") : 0;
    std::vector<size_t> offsetIndices;
    Expr offsetSel; // selection-matrix batch [-4: 1,  -3: T, -2: S, -1: N]
    auto N = offsetEmbeddingRange * 2 + 1; // we distinguish on each side at most 'offsetEmbeddingRange' positions, e.g. offsetEmbeddingRange == 2
    auto B = q->shape()[-4];
    auto H = q->shape()[-3];
    auto T = q->shape()[-2];
    auto S = k->shape()[-2];
    if (offsetEmbeddingRange)
    {
      // note: in decoding, T=1
      ABORT_IF((T != S && T != 1) || B != k->shape()[-4] || !isSelf,
               "inconsistent dimensions in self-attention S={} T={} B={} k[4]={} (offset embedding)??", S, T, B, k->shape()[-4]);
      static bool shouted = false;
      if (!shouted)
      {
        shouted = true;
        LOG(info, "offset embedding, range {}", offsetEmbeddingRange);
      }
      auto WAk = graph->param(prefix + "_WAk", {N, dk}, inits::glorot_uniform);  // [N, split vector dim]
      //WAk->debug("WAk");
      // project every q (across batch, beams, time steps) with all offset embeddings
      //auto q2d = flatten_2d(q); // [-2: beam depth * batch size * num heads * max tgt length, -1: split vector dim] flatten out the vectors across all those dimensions into a single matrix
      //q->debug("q");
      // TODO: the flattening to matrix is not needed, actually.
      //                                          q : [-4: beam depth * batch size, -3: num heads, -2: max tgt length, -1: split vector dim]
      auto qWAk = dot(q, WAk, false, true, scale); // [-4: beam depth * batch size, -3: num heads, -2: max tgt length, -1: N]
      //qWAk->debug("qWAk");
      // scatter qw[b,h,t,s] <- qw[b,h,t,clamp(s-t)]
      // Implemented as a batch matrix product. OK for small N, and OK anyway for qWAk (scalar).
      // For every t we need a NxS-dim matrix: [-4: 1,  -3: T, -2: S, -1: N]  1 for n == embedding index for (s-t)
      // repeat x BH:                          [-4: BH, -3, T: -2: S, -1: N]  left factor
      // and reshape qWAk to:                  [-4: BH, -3: T, -2: N, -1: d]  right factor, d = 1 for z (and d == dk later for v)
      // resulting bdot shape is:              [-4: BH, -3: T, -2: S, -1: d]
      // reshape back to:                      [-4: B,  -3: H, -2: T, -1: S]
      std::vector<float> offsetSelCpu(T * S * N, 0);
      auto locate = [&](size_t t, size_t s, size_t n) { return (t * S + s) * N + n; };
      for (int t = 0; t < T; t++)
        for (int s = 0; s < S; s++)
      {
        auto offset = s - (t + S - T); // the (S-T) adjusts for the inference case which calls this once per output step
        auto n = offset + offsetEmbeddingRange; // convert to index
        n = n < 0 ? 0 : n > (N-1) ? (N-1) : n;  // clamp to range
        offsetSelCpu[locate(t, s, n)] = 1;
      }
      offsetSel = graph->constant({1, T, S, N},       // [-4: 1,  -3: T, -2: S, -1: N]
                                  inits::from_vector(offsetSelCpu));
      //offsetSel->debug("offsetSel (initial)");
      offsetSel = repeat(offsetSel, B*H, axis = -4);  // [-4: BH, -3, T: -2: S, -1: N]
      qWAk = reshape(qWAk, {B*H, T, N, 1});           // [-4: BH, -3: T, -2: N, -1: d=1]
      auto zk = bdot(offsetSel, qWAk);                // [-4: BH, -3: T, -2: S, -1: d=1]
      zk = reshape(zk, {B, H, T, S});
      //zk->debug("zk");
      z = z + zk;
    }

    // take softmax along src sequence axis (-1)
    auto zm = z + mask;
    auto weights = softmax(zm); // [-4: beam depth * batch size, -3: num heads, -2: max tgt length, -1: max src length]

    // my strange experiment
    auto heads = q->shape()[-3];
    static int n = 0; // log counter
    if (pExtraLoss && heads == 1)
    {
      auto Pst = weights; // P(s|t) : [-4: beam depth * batch size, -3: num heads, -2: max tgt length, -1: max src length]
      ABORT_IF(heads != 1, "my strange experiment requires heads = 1, not {}", heads);
      static bool shouted = false;
      if (!shouted)
      {
        shouted = true;
        LOG(info, "computing extraLoss");
      }
      // compute P(t|s), which is weights normalized along time axis [-2]
      // Marian cannot taken softmax along other axes, so we must transpose
      auto zmT = transpose(zm, {0, 1, 3, 2}); // [-4: beam depth * batch size, -3: num heads, -2: max src length, -1: max tgt length]
      auto Pts = softmax(zmT); // P(t|s); softmax is along original axis [-2] (tgt sequence axis) which is now [-1]
      auto PtsT = transpose(Pts, {0, 1, 3, 2}); // P(t|s) : [-4: beam depth * batch size, -3: num heads, -2: max tgt length, -1: max src length]
      auto prod = Pst * PtsT; // elementwise product [-4: beam depth * batch size, -3: num heads, -2: max tgt length, -1: max src length]
      auto Pss = sum(prod, axis = -2); // elementwise product [-4: beam depth * batch size, -3: num heads (1), -2: 1, -1: max src length]
      auto mulMask = exp(mask * 1e8);
      auto logPss = log(Pss + 1e-8) * mulMask;
      *pExtraLoss = -sum(sum(logPss, axis = -1), axis = -4);
      if (n % 100 == 0 && graph->getDevice().no == 0)
      {
        //mulMask->debug("mulMask");
        Pts->debug("Pts");
        Pst->debug("Pst");
        Pss->debug("Pss");
        (*pExtraLoss)->debug("loss");
      }
      //weights = weights + 0 * sum(sum(Pss, axis = -1), axis = -4); // HACK: make sure it gets evaluated, so we get the debug message
    }

#if 1 // my strange experiment
    if (pSentEndProb && heads == 1)
    {
      static bool shouted = false;
      if (!shouted)
      {
        shouted = true;
        LOG(info, "sent-end prob model enabled");
      }
      auto getLast = [](Expr x, Expr mulMask, int/*keywords::axis_k*/ ax) // get last of 'x' according to (multiplicative) element-wise mask, along axis 'ax'
      {
        auto lastMask = mulMask - delay(mulMask, -1, /*axis=*/ax/*source-time direction*/); // selector for last source position (11100 --> 00100)
        //mulMask->debug("getLast: mulMask");
        //lastMask->debug("getLast: lastMask");
        return sum(x * lastMask, ax); // select last step [-4: beam depth * batch size, -3: num heads (1), -2: max tgt length, -1: 1]
        // ^^ this is actually a bdot, but of two vectors, which current bdot() cannot do
      };
      // pick out the sentence-end probability for each sequence
      // We use the mask.
      // mask : [-4: batch size,              -3: num heads broadcast=1, -2: max length broadcast=1, -1: max length]
      // all  : [-4: beam depth * batch size, -3: num heads,             -2: max tgt length,         -1: max src length]
      auto all = weights; // pSentEndProb gets assigned the respective last step's' value of this
      auto last = getLast(all, exp(mask * 1e8), /*axis=*/-1); // select last step [-4: beam depth * batch size, -3: num heads (1), -2: max tgt length, -1: 1]
      // BUGBUG: If I return logsoftmax(zm) instead, and exponentiate later, it does not converge. There is a bug somewhere.
      // note: last's shape is compatible with 'output'
      if (n % 100 == 0 && graph->getDevice().no == 0)
      {
        auto Pst = weights; // P(s|t) : [-4: beam depth * batch size, -3: num heads, -2: max tgt length, -1: max src length]
        Pst->debug("Pst");
        last->debug("last");
      }
      *pSentEndProb = last; // [-4: beam depth * batch size, -3: num heads (1), -2: max tgt length, -1: 1]
    }
#endif
    if (graph->getDevice().no == 0)
      n++;

    // optional dropout for attention weights
    // TODO: Does this really make sense? We don't drop out the final softmax either...
    float dropProb
        = inference ? 0 : options->get<float>("transformer-dropout-attention");

    if(dropProb)
      weights = dropout(weights, dropProb);

    // apply attention weights to values
    auto output = bdot(weights, v);   // [-4: beam depth * batch size, -3: num heads, -2: max tgt length, -1: split vector dim]

    // add in offset embeddings
    if (offsetEmbeddingRange)
    {
      auto WAv = graph->param(prefix + "_WAv", {N, dk}, inits::glorot_uniform);  // [N, split vector dim]
      //WAv->debug("WAv");
      // weights:          [-4: beam depth * batch size, -3: num heads, -2: max tgt length, -1: max src length]
      // reduce weights to [-4: beam depth * batch size, -3: num heads, -2: max tgt length, -1: N]
      // offsetSel:                                                   [BH, T: S, N]
      auto weights1 = reshape(weights, {B*H, T, S, 1});            // [BH, T, S, 1]
      //weights1->debug("weights1");
      auto offsetWeights = bdot(offsetSel, weights1, true, false); // [BH, T, N, 1]
      offsetWeights = reshape(offsetWeights, {B, H, T, N});        // [B,  H, T, N]
      //offsetWeights->debug("offsetWeights");
      // sum over Wav's rows, weighted by offsetWeights. This is a matrix product.
      Expr offsetOutput = dot(offsetWeights, WAv);                 // [B, H, T, dk]
      //offsetOutput = reshape(offsetOutput, {B, H, T, dk}); // [-4: beam depth * batch size, -3: num heads, -2: max tgt length, -1: split vector dim]
      output = output + offsetOutput; // [-4: beam depth * batch size, -3: num heads, -2: max tgt length, -1: split vector dim]
    }

    return output;
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
                        bool inference = false, Expr* pExtraLoss = nullptr, Expr* pSentEndProb = nullptr,
                        bool isSelf = false) {
    using namespace keywords;

    int dimModel = q->shape()[-1];

#if 1 // [fseide]
    // This projection may be important to allow multi-head attention to slice in directions.
    // But note that the subsequent MLP should be able to munge things together.
    // Without this, the multi-head split is always on the same embedding dimensions.
    const auto noQKProjection = false;
    static bool shouted = false;
    if (noQKProjection && !shouted)
    {
      fprintf(stderr,"### noQKProjection mode\n"), fflush(stderr);
      shouted = true;
    }
    auto Wq = noQKProjection ? Expr() : graph->param(
        prefix + "_Wq", {dimModel, dimModel}, inits::glorot_uniform);
    auto bq = noQKProjection ? Expr() : graph->param(prefix + "_bq", {1, dimModel}, inits::zeros);
    auto qh = noQKProjection ? q : affine(q, Wq, bq);
#else
    auto Wq = graph->param(
        prefix + "_Wq", {dimModel, dimModel}, inits::glorot_uniform);
    auto bq = graph->param(prefix + "_bq", {1, dimModel}, inits::zeros);
    auto qh = affine(q, Wq, bq);
#endif
    qh = SplitHeads(qh, dimHeads); // [-4: beam depth * batch size, -3: num heads, -2: max length, -1: split vector dim]

    std::vector<Expr> outputs;
    for(int i = 0; i < keys.size(); ++i) {
      std::string prefixProj = prefix;
      if(i > 0)
        prefixProj += "_enc" + std::to_string(i + 1);

#if 1 // [fseide]
      auto Wk = noQKProjection ? Expr() : graph->param(prefixProj + "_Wk",
                             {dimModel, dimModel},
                             inits::glorot_uniform);
      auto bk = noQKProjection ? Expr() : graph->param(
          prefixProj + "_bk", {1, dimModel}, inits::zeros);

      auto Wv = noQKProjection ? Expr() : graph->param(prefixProj + "_Wv",
                             {dimModel, dimModel},
                             inits::glorot_uniform);
      auto bv = noQKProjection ? Expr() : graph->param(
          prefixProj + "_bv", {1, dimModel}, inits::zeros);

      auto kh = noQKProjection ? keys[i]   : affine(keys[i],   Wk, bk); // [-4: beam depth, -3: batch size, -2: max length, -1: vector dim]
      auto vh = noQKProjection ? values[i] : affine(values[i], Wv, bv);
#else
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
#endif

      kh = SplitHeads(kh, dimHeads); // [-4: batch size, -3: num heads, -2: max length, -1: split vector dim]
      vh = SplitHeads(vh, dimHeads); // [-4: batch size, -3: num heads, -2: max length, -1: split vector dim]

      // apply multi-head attention to downscaled inputs
      auto output
          = Attention(graph, options, prefix, qh, kh, vh, masks[i], inference, pExtraLoss, pSentEndProb, isSelf); // [-4: beam depth * batch size, -3: num heads, -2: max length, -1: split vector dim]

      output = JoinHeads(output, q->shape()[-4]); // [-4: beam depth, -3: batch size, -2: max length, -1: vector dim]
      outputs.push_back(output);
    }

    Expr output;
    if(outputs.size() > 1)
      output = concatenate(outputs, axis = -1);
    else
      output = outputs.front();

    int dimAtt = output->shape()[-1];

    auto Wo
        = graph->param(prefix + "_Wo", {dimAtt, dimOut}, inits::glorot_uniform);
    auto bo = graph->param(prefix + "_bo", {1, dimOut}, inits::zeros);
    output = affine(output, Wo, bo);

    return output;
  }

  static Expr LayerAttention(Ptr<ExpressionGraph> graph,
                             Ptr<Options> options,
                             std::string prefix,
                             Expr input,
                             Expr keys,
                             Expr values,
                             Expr mask,
                             bool inference = false, bool isTop = false, Expr* pExtraLoss = nullptr, Expr* pSentEndProb = nullptr,
                             bool isSelf = false) {
    return LayerAttention(graph,
                          options,
                          prefix,
                          input,
                          std::vector<Expr>{keys},
                          std::vector<Expr>{values},
                          std::vector<Expr>{mask},
                          inference, isTop, pExtraLoss, pSentEndProb,
                          isSelf);
  }

  static Expr LayerAttention(Ptr<ExpressionGraph> graph,
                             Ptr<Options> options,
                             std::string prefix,
                             Expr input,                      // [-4: beam depth, -3: batch size, -2: max length, -1: vector dim]
                             const std::vector<Expr> &keys,   // [-4: beam depth=1, -3: batch size, -2: max length, -1: vector dim]
                             const std::vector<Expr> &values,
                             const std::vector<Expr> &masks,  // [-4: batch size, -3: num heads broadcast=1, -2: max length broadcast=1, -1: max length]
                             bool inference = false, bool isTop = false, Expr* pExtraLoss = nullptr, Expr* pSentEndProb = nullptr,
                             bool isSelf = false) {
    using namespace keywords;

    int dimModel = input->shape()[-1];

    float dropProb = inference ? 0 : options->get<float>("transformer-dropout");
    auto opsPre = options->get<std::string>("transformer-preprocess");
    auto output = PreProcess(graph, prefix + "_Wo", opsPre, input, dropProb);

#if 1
    bool hasTopHeads = isTop && options->has("transformer-heads-top");
    auto heads = hasTopHeads ? options->get<int>("transformer-heads-top") : options->get<int>("transformer-heads"); 
#if 1 // BUGBUG: some models got screwed up
    if (options->get<int>("transformer-heads") == 16 && heads == 8)
    {
      heads = 16; // I believe models that report this combination actually used 16 (have better decoding result)
      static bool shouted = false;
      if (!shouted)
      {
        LOG(info, "BUGBUG workaround: top target-source-attentionheads corrected back to {}", heads);
        shouted = true;
      }
    }
#endif
    if (hasTopHeads)
    {
      static bool shouted = false;
      if (!shouted)
      {
        LOG(info, "top target-source-attentionheads = {}", heads);
        shouted = true;
      }
    }
    if (!hasTopHeads || heads != 1)
      pExtraLoss = nullptr; // clear this if not my strange experiment
#else
    auto heads = options->get<int>("transformer-heads");
#endif

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
                       inference, pExtraLoss, pSentEndProb,
                       isSelf);

    auto opsPost = options->get<std::string>("transformer-postprocess");
    output
        = PostProcess(graph, prefix + "_Wo", opsPost, output, input, dropProb);

    return output;
  }

  static Expr LayerFFN(Ptr<ExpressionGraph> graph,
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

    auto W1 = graph->param(
        prefix + "_W1", {dimModel, dimFfn}, inits::glorot_uniform);
    auto b1 = graph->param(prefix + "_b1", {1, dimFfn}, inits::zeros);

    auto W2 = graph->param(
        prefix + "_W2", {dimFfn, dimModel}, inits::glorot_uniform);
    auto b2 = graph->param(prefix + "_b2", {1, dimModel}, inits::zeros);

    output = affine(output, W1, b1);
    if(options->get<std::string>("transformer-ffn-activation") == "relu")
      output = relu(output);
    else
      output = swish(output);

    float ffnDropProb
        = inference ? 0 : options->get<float>("transformer-dropout-ffn");
    if(ffnDropProb)
      output = dropout(output, ffnDropProb);

    output = affine(output, W2, b2);

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

    // [fseide]
    const auto crossLayerAttention = false;
    std::vector<Expr> allLayersContexts;
    static bool shouted = false;
    if (crossLayerAttention && !shouted)
    {
      fprintf(stderr,"### crossLayerAttention mode\n"), fflush(stderr);
      shouted = true;
    }

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
                             inference_,
                             /*isTop=*/false, /*pExtraLoss=*/nullptr, /*pSentEndProb=*/nullptr,
                             /*isSelf=*/true);

      layer = LayerFFN(graph,
                       options_,
                       prefix_ + "_l" + std::to_string(i) + "_ffn",
                       layer,
                       inference_);

      if (crossLayerAttention)
          allLayersContexts.push_back(layer); // [-4: beam depth, -3: batch size, -2: max length, -1: vector dim]
    }

    // to do attention over all layers jointly, concatenate them all
    // In Transformer, these tensors are sets, not sequences, so we can just concat them in "time" axis.
    if (crossLayerAttention)
    {
        layer = concatenate(allLayersContexts, axis = -2);                  // [-4: beam depth, -3: batch size, -2: max length * N, -1: vector dim]
        batchMask = repeat(batchMask, allLayersContexts.size(), axis = -3); // [-4: beam depth=1, -3: max length * N, -2: batch size, -1: vector dim=1]
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
                   std::vector<Ptr<EncoderState>> &encStates)
      : DecoderState(states, probs, encStates) {}
  TransformerState(const rnn::States &states,
                   Expr probs, Expr extraLoss,
                   std::vector<Ptr<EncoderState>> &encStates)
      : DecoderState(states, probs, extraLoss, encStates) {}

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

    return New<TransformerState>(selectedStates, probs_, encStates_);
  }
};

class DecoderTransformer : public DecoderBase, public Transformer {
public:
  DecoderTransformer(Ptr<Options> options) : DecoderBase(options) {}

  virtual Ptr<DecoderState> startState( // TODO: const?
      Ptr<ExpressionGraph> graph,
      Ptr<data::CorpusBatch> batch,
      std::vector<Ptr<EncoderState>> &encStates) {
    rnn::States startStates;
    return New<TransformerState>(startStates, nullptr, encStates);
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

    int startPos = 0;
    auto prevDecoderStates = state->getStates();
    if(prevDecoderStates.size() > 0)
      startPos = prevDecoderStates[0].output->shape()[-2];

    scaledEmbeddings
        = AddPositionalEmbeddings(graph, scaledEmbeddings, startPos);

    scaledEmbeddings = atleast_nd(scaledEmbeddings, 4);

    // reorganize batch and timestep
    auto query = TransposeTimeBatch(scaledEmbeddings); // [-4: beam depth=1, -3: batch size, -2: max length, -1: vector dim]

    auto opsEmb = opt<std::string>("transformer-postprocess-emb");
    float dropProb = inference_ ? 0 : opt<float>("transformer-dropout");

    query = PreProcess(graph, prefix_ + "_emb", opsEmb, query, dropProb);

    rnn::States decoderStates;
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

    selfMask = transposedLogMask(selfMask);

    // reorganize batch and timestep for encoder embeddings
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

    // apply decoder layers
    Expr extraLoss, sentEndProb; // (my strange experiment)
    auto decDepth = opt<int>("dec-depth");
    for(int i = 1; i <= decDepth; ++i) {
      auto isTop = i == decDepth;
      auto values = query;
      if(prevDecoderStates.size() > 0)
        values
            = concatenate({prevDecoderStates[i - 1].output, query}, axis = -2);

      decoderStates.push_back({values, nullptr});

      // TODO: do not recompute matrix multiplies
      // self-attention
      query = LayerAttention(graph,
                             options_,
                             prefix_ + "_l" + std::to_string(i) + "_self",
                             query,
                             values,
                             values,
                             selfMask,
                             inference_,
                             /*isTop=*/false, /*pExtraLoss=*/nullptr, /*pSentEndProb=*/nullptr,
                             /*isSelf=*/true);

      // attention over encoder
      if(encoderContexts.size() > 0) {
        // auto comb = opt<std::string>("transformer-multi-encoder");
        std::string comb = "stack";
        if(comb == "concat") {
          query
              = LayerAttention(graph,
                               options_,
                               prefix_ + "_l" + std::to_string(i) + "_context",
                               query,
                               encoderContexts,
                               encoderContexts,
                               encoderMasks,
                               inference_, isTop, nullptr/*&extraLoss*/, &sentEndProb);

        } else if(comb == "stack") {
          for(int j = 0; j < encoderContexts.size(); ++j) { // multiple encoders are applied one after another
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
                                   inference_, isTop, nullptr/*&extraLoss*/, &sentEndProb);
          }
        } else {
          ABORT("Unknown value for transformer-multi-encoder: {}", comb);
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

    int dimTrgVoc = opt<std::vector<int>>("dim-vocabs")[batchIndex_];

    auto layerOut = mlp::dense(graph)          //
        ("prefix", prefix_ + "_ff_logit_out")  //
        ("dim", dimTrgVoc);

    if(opt<bool>("tied-embeddings") || opt<bool>("tied-embeddings-all")) {
      std::string tiedPrefix = prefix_ + "_Wemb";
      if(opt<bool>("tied-embeddings-all") || opt<bool>("tied-embeddings-src"))
        tiedPrefix = "Wemb";
      layerOut.tie_transposed("W", tiedPrefix);
    }

    // assemble layers into MLP and apply to embeddings, decoder context and
    // aligned source context
    auto output = mlp::mlp(graph).push_back(layerOut);

    Expr logits = output->apply(decoderContext); // [-4: beam depth=1, -3: max length, -2: batch size, -1: vocab dim]

#if 1
    if (sentEndProb)
    {
      auto where = [](Expr selector, Expr oneVal, Expr zeroVal) // return oneVal where selector=1, else zeroVal. Cf. numpy.where(), cntk.element_select()
      {
        return selector * oneVal + (1-selector) * zeroVal;
      };
      // this mode:
      //  - P(y|...) = select (u_end, P(end|...),(1-P(end|...)) * softmax_noEnd(f(context_vector))
      //             ~ select (u_end, odds(end|...), softmax_noEnd(f(context_vector))
      //  - log P (y|...) = select (u_end, log odds(end|...), log softmax_noEnd(f(context_vector))
      //  - The context vector should ideally not look at the end position, but that requires renormalization.
      //    Maybe OK since this is a hack anyway.
      auto Pend = sentEndProb;  // [-4: beam depth * batch size, -3: 1, -2: max tgt length, -1: 1]
      Pend = reshape(Pend, { query->shape()[-4], query->shape()[-3], query->shape()[-2], 1 }); // [-4: beam depth, -3: batch size, -2: max length, -1: 1]
      Pend = TransposeTimeBatch(Pend); // [-4: beam depth, -3: max length, -2: batch size, -1: 1]
      std::vector<float> u_EOSCpu(dimTrgVoc, 0);
      u_EOSCpu[EOS_ID] = 1;
      auto u_EOS = graph->constant({1, 1, 1, dimTrgVoc}, // [-4: 1,          -3: 1,          -2: 1,          -1: vocab dim]
                                   inits::from_vector(u_EOSCpu));
      auto logits_notEOS = logits + -99999999.f * u_EOS; // [-4: beam depth, -3: max length, -2: batch size, -1: vocab dim]
      logits = where(u_EOS,
                     log(Pend + std::numeric_limits<float>::min()) - log((1-Pend) + std::numeric_limits<float>::min()),
                     logsoftmax(logits_notEOS));
      auto targetMask = state->getTargetMask();
      if (targetMask)
      {
        targetMask = atleast_4d(targetMask);      // [1, max length, batch size, 1] --TODO: should works without this; try, then remove
        logits = logits * targetMask; // suppress tokens beyond end of target for good measure (not sure if still needed)
      }
      static int n = 0;
      if (n % 100 == 0 && graph->getDevice().no == 0)
      {
        LOG(info, "snapshot {}", n);
        Pend->debug("Pend");
        logits->debug("logits");
      }
      if (graph->getDevice().no == 0)
        n++;
    }
#endif

    // return unormalized(!) probabilities
#if 1 // (my strange experiment)
    if (extraLoss)
    {
      static bool shouted = false;
      if (!shouted)
      {
        shouted = true;
        LOG(info, "got extraLoss, shape is {}", extraLoss->shape());
      }
      return New<TransformerState>(
        decoderStates, logits, extraLoss, state->getEncoderStates());
    }
#endif
    return New<TransformerState>(
        decoderStates, logits, state->getEncoderStates());
  }

  // helper function for guided alignment
  virtual const std::vector<Expr> getAlignments(int i = 0) { return {}; }

  void clear() {}
};
}
