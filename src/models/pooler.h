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

  virtual Expr apply(Ptr<ExpressionGraph>, Ptr<data::CorpusBatch>, const std::vector<Ptr<EncoderState>>&) = 0;

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

  Expr apply(Ptr<ExpressionGraph> graph, Ptr<data::CorpusBatch> batch, const std::vector<Ptr<EncoderState>>& encoderStates) override {
    ABORT_IF(encoderStates.size() != 1, "Pooler expects exactly one encoder state");

    auto context = encoderStates[0]->getContext();
    auto batchMask = encoderStates[0]->getMask();

    // do a max pool here
    Expr logMask = (1.f - batchMask) * -9999.f;
    Expr maxPool = max(context * batchMask + logMask, /*axis=*/-3);

    return maxPool;
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

  Expr apply(Ptr<ExpressionGraph> graph, Ptr<data::CorpusBatch> batch, const std::vector<Ptr<EncoderState>>& encoderStates) override {
    ABORT_IF(encoderStates.size() != 1, "Pooler expects exactly one encoder state");

    auto context = encoderStates[0]->getContext();
    auto batchMask = encoderStates[0]->getMask();

    // Corresponds to the way we do this in transformer.h
    // @TODO: unify this better, this is currently hacky
    Expr slicePool = slice(context * batchMask, /*axis=*/-3, 0);

    return slicePool;
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

  Expr apply(Ptr<ExpressionGraph> graph, Ptr<data::CorpusBatch> batch, const std::vector<Ptr<EncoderState>>& encoderStates) override {
    ABORT_IF(encoderStates.size() != 2, "SimPooler expects exactly two encoder states");

    std::vector<Expr> vecs;
    for(auto encoderState : encoderStates) {
      auto context = encoderState->getContext();
      auto batchMask = encoderState->getMask();

      Expr pool;
      auto type = options_->get<std::string>("original-type");
      if(type == "laser") {
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

    auto scalars = scalar_product(vecs[0], vecs[1], /*axis*/-1);
    auto length1 = sqrt(sum(square(vecs[0]), /*axis=*/-1));
    auto length2 = sqrt(sum(square(vecs[1]), /*axis=*/-1));

    auto cosine  = scalars / ( length1 * length2 );

    return cosine;
  }

  void clear() override {}

};

}