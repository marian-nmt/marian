#pragma once

#include "layers/generic.h"
#include "layers/guided_alignment.h"
#include "layers/loss.h"
#include "layers/weight.h"
#include "models/encoder_decoder.h"
#include "models/encoder_classifier.h"

namespace marian {
namespace models {

// @TODO: this whole file is an unholy mess and needs to be refactored.
// Using MultiRationalLoss is a first improvement, but we can probably
// unify classifier and decoder costs. Also rethink step-wise cost.

// @TODO: inheritance and polymorphism is used here in a rather unclear way.
// E.g. returns Ptr<MultiRationalLoss> which should be Ptr<RationalLoss>?
// Other functions return RationalLoss directly without Ptr<...>, but also
// they do not need polymorphism here.

class CostBase {
public:
  virtual Ptr<MultiRationalLoss> apply(Ptr<ModelBase> model,
                                       Ptr<ExpressionGraph> graph,
                                       Ptr<data::Batch> batch,
                                       bool clearGraph = true) = 0;
};

class EncoderDecoderCE : public CostBase {
protected:
  Ptr<Options> options_;

  bool inference_{false};
  bool toBeWeighted_{false};

  // @TODO: single loss seems wrong
  Ptr<LabelwiseLoss> loss_;
  Ptr<WeightingBase> weighter_;

public:
  EncoderDecoderCE(Ptr<Options> options)
      : options_(options), inference_(options->get<bool>("inference", false)) {
    loss_ = newLoss(options_, inference_);

    toBeWeighted_
        = (options_->hasAndNotEmpty("data-weighting") && !inference_)
          || (options_->has("dynamic-weighting") && options_->get<bool>("dynamic-weighting")
              && !inference_);
    if(toBeWeighted_)
      weighter_ = WeightingFactory(options_);
  }

  Ptr<MultiRationalLoss> apply(Ptr<ModelBase> model,
             Ptr<ExpressionGraph> graph,
             Ptr<data::Batch> batch,
             bool clearGraph = true) override {
    auto encdec = std::static_pointer_cast<EncoderDecoder>(model);
    auto corpusBatch = std::static_pointer_cast<data::CorpusBatch>(batch);

    auto state = encdec->stepAll(graph, corpusBatch, clearGraph);

    Expr weights;
    if(toBeWeighted_)
      weights = weighter_->getWeights(graph, corpusBatch);

    // multi-objective training
    Ptr<MultiRationalLoss> multiLoss = newMultiLoss(options_);

    // @TODO: adapt to multi-objective training with multiple decoders
    auto partialLoss = loss_->apply(state->getLogProbs(),
                                    state->getTargetIndices(),
                                    state->getTargetMask(),
                                    weights);
    multiLoss->push_back(partialLoss);

    if(options_->get("guided-alignment", std::string("none")) != "none" && !inference_) {
      auto attentionVectors = encdec->getDecoders()[0]->getAlignments();
      ABORT_IF(attentionVectors.empty(), "Model does not seem to support alignments");

      auto attention = concatenate(attentionVectors, /*axis =*/ -1);

      auto alignmentLoss = guidedAlignmentCost(graph, corpusBatch, options_, attention);
      multiLoss->push_back(alignmentLoss);
    }

    return multiLoss;
  }
};

// Wraps an EncoderClassifier so it can produce a cost from raw logits. @TODO: Needs refactoring
class EncoderClassifierCE : public CostBase {
protected:
  Ptr<Options> options_;
  bool inference_{false};

  // @TODO: single loss seems wrong, especially since we support multiple objectives here,
  // also not sure this needs to be a member at all.
  Ptr<LabelwiseLoss> loss_;

public:
  EncoderClassifierCE(Ptr<Options> options)
      : options_(options), inference_(options->get<bool>("inference", false)) {
    loss_ = newLoss(options_, inference_);
  }

  Ptr<MultiRationalLoss> apply(Ptr<ModelBase> model,
             Ptr<ExpressionGraph> graph,
             Ptr<data::Batch> batch,
             bool clearGraph = true) override {

    auto enccls = std::static_pointer_cast<EncoderClassifier>(model);
    auto corpusBatch = std::static_pointer_cast<data::CorpusBatch>(batch);

    auto states = enccls->apply(graph, corpusBatch, clearGraph);

    // multi-objective training
    Ptr<MultiRationalLoss> multiLoss = newMultiLoss(options_);
    for(int i = 0; i < states.size(); ++i) {
      auto partialLoss = loss_->apply(states[i]->getLogProbs(),
                                      states[i]->getTargetIndices(),
                                      /*mask=*/nullptr,
                                      /*weights=*/nullptr);
      multiLoss->push_back(partialLoss);
    }
    return multiLoss;
  }
};

class Trainer : public ModelBase {
protected:
  Ptr<ModelBase> model_;
  Ptr<CostBase> cost_;

public:
  Trainer(Ptr<ModelBase> model, Ptr<CostBase> cost)
      : model_(model), cost_(cost) {}

  Ptr<ModelBase> getModel() { return model_; }

  virtual void load(Ptr<ExpressionGraph> graph,
                    const std::string& name,
                    bool markedReloaded = true) override {
    model_->load(graph, name, markedReloaded);
  };

  virtual void save(Ptr<ExpressionGraph> graph,
                    const std::string& name,
                    bool saveTranslatorConfig = false) override {
    model_->save(graph, name, saveTranslatorConfig);
  }

  virtual Ptr<RationalLoss> build(Ptr<ExpressionGraph> graph,
                                  Ptr<data::Batch> batch,
                                  bool clearGraph = true) override {
    return cost_->apply(model_, graph, batch, clearGraph);
  };

  virtual void clear(Ptr<ExpressionGraph> graph) override { model_->clear(graph); };
};

typedef Trainer Scorer;

class CostStep {
public:
  virtual Ptr<DecoderState> apply(Ptr<DecoderState> state) = 0;
};

class LogSoftmaxStep : public CostStep {
public:
  virtual Ptr<DecoderState> apply(Ptr<DecoderState> state) override {
    // decoder needs normalized probabilities (note: skipped if beam 1 and --skip-cost)
    auto logits = state->getLogProbs();

    auto logprobs = logsoftmax(logits);

    state->setLogProbs(logprobs);
    return state;
  }
};

// Gumbel-max noising for sampling during beam-search
// Seems to work well enough with beam-size=1. Turn on
// with --output-sampling during translation with marian-decoder
class GumbelSoftmaxStep : public CostStep {
public:
  virtual Ptr<DecoderState> apply(Ptr<DecoderState> state) override {
    auto logits = state->getLogProbs();

    auto logprobs = logsoftmax(logits + constant_like(logits, inits::gumbel));

    state->setLogProbs(logprobs);
    return state;
  }
};

// class to wrap an EncoderDecoderBase and a CostStep that are executed in sequence,
// wrapped again in the EncoderDecoderBase interface
// @TODO: seems we are conflating an interface defition with its implementation?
class Stepwise : public EncoderDecoderBase {
protected:
  Ptr<EncoderDecoderBase> encdec_;
  Ptr<CostStep> cost_;

public:
  Stepwise(Ptr<EncoderDecoderBase> encdec, Ptr<CostStep> cost)
      : encdec_(encdec), cost_(cost) {}

  virtual void load(Ptr<ExpressionGraph> graph,
                    const std::string& name,
                    bool markedReloaded = true) override {
    encdec_->load(graph, name, markedReloaded);
  }

  virtual void mmap(Ptr<ExpressionGraph> graph,
                    const void* ptr,
                    bool markedReloaded = true) override {
    encdec_->mmap(graph, ptr, markedReloaded);
  };

  virtual void save(Ptr<ExpressionGraph> graph,
                    const std::string& name,
                    bool saveTranslatorConfig = false) override {
    encdec_->save(graph, name, saveTranslatorConfig);
  }

  virtual void clear(Ptr<ExpressionGraph> graph) override { encdec_->clear(graph); }

  virtual Ptr<RationalLoss> build(Ptr<ExpressionGraph> graph,
                                  Ptr<data::Batch> batch,
                                  bool clearGraph = true) override {
    auto corpusBatch = std::static_pointer_cast<data::CorpusBatch>(batch);
    return build(graph, corpusBatch, clearGraph);
  }

  virtual Ptr<DecoderState> startState(Ptr<ExpressionGraph> graph,
                                       Ptr<data::CorpusBatch> batch) override {
    return encdec_->startState(graph, batch);
  }

  virtual Ptr<DecoderState> step(Ptr<ExpressionGraph> graph,
                                 Ptr<DecoderState> state,
                                 const std::vector<IndexType>& hypIndices,
                                 const std::vector<IndexType>& embIndices,
                                 int dimBatch,
                                 int beamSize) override {
    auto nextState = encdec_->step(
        graph, state, hypIndices, embIndices, dimBatch, beamSize);
    return cost_->apply(nextState);
  }

  virtual Ptr<RationalLoss> build(Ptr<ExpressionGraph> /*graph*/,
                                  Ptr<data::CorpusBatch> /*batch*/,
                                  bool /*clearGraph*/ = true) override {
    ABORT("Wrong wrapper. Use models::Trainer or models::Scorer");
    return nullptr;
  }

  virtual Ptr<Options> getOptions() override { return encdec_->getOptions(); };

  virtual void setShortlistGenerator(
      Ptr<data::ShortlistGenerator> shortlistGenerator) override {
    encdec_->setShortlistGenerator(shortlistGenerator);
  };

  virtual Ptr<data::Shortlist> getShortlist() override {
    return encdec_->getShortlist();
  };

  virtual data::SoftAlignment getAlignment() override { return encdec_->getAlignment(); }
};

inline Ptr<ModelBase> add_cost(Ptr<EncoderDecoder> encdec,
                               Ptr<Options> options) {
  switch(options->get<usage>("usage", usage::raw)) {
    case usage::training:
      return New<Trainer>(encdec, New<EncoderDecoderCE>(options));
    case usage::scoring:
      return New<Scorer>(encdec, New<EncoderDecoderCE>(options));
    case usage::translation:
      if(options->get<bool>("output-sampling", false))
        return New<Stepwise>(encdec, New<GumbelSoftmaxStep>());
      else
        return New<Stepwise>(encdec, New<LogSoftmaxStep>());
    case usage::raw:
    default:
      return encdec;
  }
}

inline Ptr<ModelBase> add_cost(Ptr<EncoderClassifier> enccls,
                               Ptr<Options> options) {
  switch(options->get<usage>("usage", usage::raw)) {
    case usage::training:
      return New<Trainer>(enccls, New<EncoderClassifierCE>(options));
    case usage::scoring:
      return New<Scorer>(enccls, New<EncoderClassifierCE>(options));
    case usage::translation:
      ABORT("Classifier cannot be used for translation");
    case usage::raw:
    default:
      return enccls;
  }
}

}  // namespace models
}  // namespace marian
