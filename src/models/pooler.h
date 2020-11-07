#pragma once

#include "marian.h"
#include "models/states.h"
#include "layers/constructors.h"
#include "layers/factory.h"

namespace marian {

/**
 * Simple base class for Poolers to be used in EncoderPooler framework
 * A pooler takes a encoder state (contextual word embeddings) and produces 
 * a single sentence embedding.
 */
class PoolerBase : public LayerBase {
  using LayerBase::LayerBase;

protected:
  const std::string prefix_{"pooler"};
  const bool inference_{false};
  const size_t batchIndex_{0};

public:
  PoolerBase(Ptr<ExpressionGraph> graph, Ptr<Options> options)
      : LayerBase(graph, options),
        prefix_(options->get<std::string>("prefix", "pooler")),
        inference_(options->get<bool>("inference", true)),
        batchIndex_(options->get<size_t>("index", 1)) {} // assume that training input has batch index 0 and labels has 1

  virtual ~PoolerBase() {}

  virtual std::vector<Expr> apply(Ptr<ExpressionGraph>, Ptr<data::CorpusBatch>, const std::vector<Ptr<EncoderState>>&) = 0;

  template <typename T>
  T opt(const std::string& key) const {
    return options_->get<T>(key);
  }

  // Should be used to clear any batch-wise temporary objects if present
  virtual void clear() = 0;
};

/**
 * Pool encoder state (contextual word embeddings) via max-pooling along sentence-length dimension.
 */
class MaxPooler : public PoolerBase {
public:
  MaxPooler(Ptr<ExpressionGraph> graph, Ptr<Options> options)
  : PoolerBase(graph, options) {}

  std::vector<Expr> apply(Ptr<ExpressionGraph> graph, Ptr<data::CorpusBatch> batch, const std::vector<Ptr<EncoderState>>& encoderStates) override {
    ABORT_IF(encoderStates.size() != 1, "Pooler expects exactly one encoder state");

    auto context = encoderStates[0]->getContext();
    auto batchMask = encoderStates[0]->getMask();

    // do a max pool here
    Expr logMask = (1.f - batchMask) * -9999.f;
    Expr maxPool = max(context * batchMask + logMask, /*axis=*/-3);

    return {maxPool};
  }

  void clear() override {}

};

/**
 * Pool encoder state (contextual word embeddings) by selecting 1st embedding along sentence-length dimension.
 */
class SlicePooler : public PoolerBase {
public:
  SlicePooler(Ptr<ExpressionGraph> graph, Ptr<Options> options)
  : PoolerBase(graph, options) {}

  std::vector<Expr> apply(Ptr<ExpressionGraph> graph, Ptr<data::CorpusBatch> batch, const std::vector<Ptr<EncoderState>>& encoderStates) override {
    ABORT_IF(encoderStates.size() != 1, "Pooler expects exactly one encoder state");

    auto context = encoderStates[0]->getContext();
    auto batchMask = encoderStates[0]->getMask();

    // Corresponds to the way we do this in transformer.h
    // @TODO: unify this better, this is currently hacky
    Expr slicePool = slice(context * batchMask, /*axis=*/-3, 0);

    return {slicePool};
  }

  void clear() override {}

};

/**
 * Not really a pooler but abusing the interface to compute a similarity of two pooled states
 */
class SimPooler : public PoolerBase {
public:
  SimPooler(Ptr<ExpressionGraph> graph, Ptr<Options> options)
  : PoolerBase(graph, options) {}

  std::vector<Expr> apply(Ptr<ExpressionGraph> graph, Ptr<data::CorpusBatch> batch, const std::vector<Ptr<EncoderState>>& encoderStates) override {
    ABORT_IF(encoderStates.size() < 2, "SimPooler expects at least two encoder states not {}", encoderStates.size());

    std::vector<Expr> vecs;
    for(auto encoderState : encoderStates) {
      auto context = encoderState->getContext();
      auto batchMask = encoderState->getMask();

      Expr pool;
      auto type = options_->get<std::string>("original-type");
      if(type == "laser" || type == "laser-sim") {
        // LASER models do a max pool here
        Expr logMask = (1.f - batchMask) * -9999.f;
        pool         = max(context * batchMask + logMask, /*axis=*/-3);
      } else if(type == "transformer") { 
        // Our own implementation in transformer.h uses a slice of the first element
        pool         = slice(context, -3, 0);
      } else {
        // @TODO: make SimPooler take Pooler objects as arguments then it won't need to know this.
        ABORT("Don't know what type of pooler to use for model type {}", type);
      }
      vecs.push_back(pool);
    }

    std::vector<Expr> outputs;
    bool trainRank = options_->hasAndNotEmpty("train-embedder-rank");

    if(!trainRank) { // inference, compute one cosine similarity only
      ABORT_IF(vecs.size() != 2, "We are expecting two inputs for similarity computation");

      // efficiently compute vector length with bdot
      auto vnorm = [](Expr e) {
        int dimModel = e->shape()[-1];
        int dimBatch = e->shape()[-2];
        e = reshape(e, {dimBatch, 1, dimModel});
        return reshape(sqrt(bdot(e, e, false, true)), {dimBatch, 1});
      };

      auto dotProduct = scalar_product(vecs[0], vecs[1], /*axis*/-1);
      auto length0 = vnorm(vecs[0]); // will be hashed and reused in the graph
      auto length1 = vnorm(vecs[1]);
      auto cosine = dotProduct / ( length0 * length1 );
      cosine = maximum(0, cosine); // clip to [0, 1] - should we actually do that?
      outputs.push_back(cosine);
    } else { // compute outputs for embedding similarity ranking
      if(vecs.size() == 2) { // implies we are sampling negative examples from the batch, since otherwise there is nothing to train
        LOG_ONCE(info, "Sampling negative examples from batch");

        auto src = vecs[0];
        auto trg = vecs[1];

        int dimModel = src->shape()[-1];
        int dimBatch = src->shape()[-2];

        src = reshape(src, {dimBatch, dimModel});
        trg = reshape(trg, {dimBatch, dimModel});

        // compute cosines between every batch entry, this produces the whole dimBatch x dimBatch matrix
        auto dotProduct = dot(src, trg, false, true); // [dimBatch, dimBatch] - computes dot product matrix
        
        auto positiveMask = dotProduct->graph()->constant({dimBatch, dimBatch}, inits::eye()); // a mask for the diagonal (positive examples are on the diagonal)
        auto negativeMask = 1.f - positiveMask; // mask for all negative examples;
        
        auto positive = sum(dotProduct * positiveMask, /*axis=*/-1); // we sum across last dim in order to get a column vector of positve examples (everything else is zero)
        outputs.push_back(positive);

        auto negative1 = dotProduct * negativeMask; // get negative examples for src -> trg (in a row)
        outputs.push_back(negative1);

        auto negative2 = transpose(negative1);  // get negative examples for trg -> src via transpose so they are located in a row
        outputs.push_back(transpose(negative2));
      } else {
        LOG_ONCE(info, "Using provided {} negative examples", vecs.size() - 2);

        // For inference and training with given set of negative examples provided in additional streams.
        // Assuming that enc0 is query, enc1 is positive example and remaining encoders are optional negative examples. Here we only use column vectors [dimBatch, 1]
        auto positive = scalar_product(vecs[0], vecs[1], /*axis*/-1);
        outputs.push_back(positive); // first tensor contains similarity between anchor and positive example

        std::vector<Expr> dotProductsNegative1, dotProductsNegative2;
        for(int i = 2; i < vecs.size(); ++i) {
          // compute similarity with anchor
          auto negative1 = scalar_product(vecs[0], vecs[i], /*axis*/-1);
          dotProductsNegative1.push_back(negative1);
          
          // for negative examples also add symmetric dot product with the positive example
          auto negative2 = scalar_product(vecs[1], vecs[i], /*axis*/-1);
          dotProductsNegative2.push_back(negative2);
        }
        auto negative1 = concatenate(dotProductsNegative1, /*axis=*/-1);
        outputs.push_back(negative1); // second tensor contains similarities between anchor and all negative example

        auto negative2 = concatenate(dotProductsNegative2, /*axis=*/-1);
        outputs.push_back(negative2); // third tensor contains similarities between positive and all negative example (symmetric)
      }
    }

    return outputs;
  }

  void clear() override {}

};

}