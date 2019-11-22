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

class ICost {
public:
  virtual Ptr<MultiRationalLoss> apply(Ptr<IModel> model,
                                       Ptr<ExpressionGraph> graph, // @TODO: why needed? Can it be gotten from model?
                                       Ptr<data::Batch> batch,
                                       bool clearGraph = true) = 0;
};

class EncoderDecoderCECost : public ICost {
protected:
  Ptr<Options> options_;

  const bool inference_{false};
  /*const*/ bool toBeWeighted_{false};

  // @TODO: single loss seems wrong
  Ptr<LabelwiseLoss> loss_;
  Ptr<WeightingBase> weighter_;

public:
  EncoderDecoderCECost(Ptr<Options> options)
      : options_(options), inference_(options->get<bool>("inference", false)) {
    loss_ = newLoss(options_, inference_);

    toBeWeighted_
        = (options_->hasAndNotEmpty("data-weighting") && !inference_)
          || (options_->has("dynamic-weighting") && options_->get<bool>("dynamic-weighting")
              && !inference_);
    if(toBeWeighted_)
      weighter_ = WeightingFactory(options_);
  }

  Ptr<MultiRationalLoss> apply(Ptr<IModel> model,
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
                                    state->getTargetWords(),
                                    state->getTargetMask(),
                                    weights);
    multiLoss->push_back(partialLoss);

    if(options_->get("guided-alignment", std::string("none")) != "none" && !inference_) {
      auto attentionVectors = encdec->getDecoders()[0]->getAlignments(); // [tgt index][beam depth, max src length, batch size, 1]
      ABORT_IF(attentionVectors.empty(), "Model does not seem to support alignments");

      auto attention = concatenate(attentionVectors, /*axis =*/ -1);

      auto alignmentLoss = guidedAlignmentCost(graph, corpusBatch, options_, attention);
      multiLoss->push_back(alignmentLoss);
    }

    return multiLoss;
  }
};

// Wraps an EncoderClassifier so it can produce a cost from raw logits. @TODO: Needs refactoring
class EncoderClassifierCECost : public ICost {
protected:
  Ptr<Options> options_;
  const bool inference_{false};

  // @TODO: single loss seems wrong, especially since we support multiple objectives here,
  // also not sure this needs to be a member at all.
  Ptr<LabelwiseLoss> loss_;

public:
  EncoderClassifierCECost(Ptr<Options> options)
      : options_(options), inference_(options->get<bool>("inference", false)) {
    loss_ = newLoss(options_, inference_);
  }

  Ptr<MultiRationalLoss> apply(Ptr<IModel> model,
             Ptr<ExpressionGraph> graph,
             Ptr<data::Batch> batch,
             bool clearGraph = true) override {

    auto enccls = std::static_pointer_cast<EncoderClassifier>(model);
    auto corpusBatch = std::static_pointer_cast<data::CorpusBatch>(batch);

    auto states = enccls->apply(graph, corpusBatch, clearGraph);

    // multi-objective training
    Ptr<MultiRationalLoss> multiLoss = newMultiLoss(options_);
    for(int i = 0; i < states.size(); ++i) {
      auto partialLoss = loss_->apply(Logits(states[i]->getLogProbs()),
                                      states[i]->getTargetWords(),
                                      /*mask=*/nullptr,
                                      /*weights=*/nullptr);
      multiLoss->push_back(partialLoss);
    }
    return multiLoss;
  }
};

class Trainer : public ICriterionFunction {
protected:
  Ptr<IModel> model_;
  Ptr<ICost> cost_;

public:
  Trainer(Ptr<IModel> model, Ptr<ICost> cost)
      : model_(model), cost_(cost) {}

  Ptr<IModel> getModel() { return model_; }

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

class ILogProb {
public:
  virtual Logits apply(Ptr<IModel> model,
                     Ptr<ExpressionGraph> graph,
                     Ptr<data::Batch> batch,
                     bool clearGraph = true) = 0;
};

// @TODO: Name 'scorer' is ambiguous: Does it compute scores for all classes, or the loss value for the ground truth?
//        Beam search uses it for the former meaning, while 'marian score' and validation in the latter.
//        This class is for the former use. The latter is done using Trainer.
class Scorer : public IModel {
protected:
  Ptr<IModel> model_;
  Ptr<ILogProb> logProb_;

public:
  Scorer(Ptr<IModel> model, Ptr<ILogProb> cost)
      : model_(model), logProb_(cost) {}

  Ptr<IModel> getModel() { return model_; }

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

  virtual Logits build(Ptr<ExpressionGraph> graph,
                     Ptr<data::Batch> batch,
                     bool clearGraph = true) override {
    return logProb_->apply(model_, graph, batch, clearGraph);
  };

  virtual void clear(Ptr<ExpressionGraph> graph) override { model_->clear(graph); };
};

class ILogProbStep {
public:
  // @BUGBUG: This is not a function application. Rather, it updates 'state' in-place.
  // Suggest to call it updateState, and not return the state object.
  virtual Ptr<DecoderState> apply(Ptr<DecoderState> state) = 0;
};

class LogSoftmaxStep : public ILogProbStep {
public:
  virtual Ptr<DecoderState> apply(Ptr<DecoderState> state) override {
    // decoder needs normalized probabilities (note: skipped if beam 1 and --skip-cost)
    state->setLogProbs(state->getLogProbs().applyUnaryFunction(logsoftmax));
    // @TODO: This is becoming more and more opaque ^^. Can we simplify this?
    return state;
  }
};

// Gumbel-max noising for sampling during beam-search
// Seems to work well enough with beam-size=1. Turn on
// with --output-sampling during translation with marian-decoder
class GumbelSoftmaxStep : public ILogProbStep {
public:
  virtual Ptr<DecoderState> apply(Ptr<DecoderState> state) override {
    state->setLogProbs(state->getLogProbs().applyUnaryFunctions(
      [](Expr logits){ // lemma gets gumbelled
        return logsoftmax(logits + constant_like(logits, inits::gumbel()));
      },
      logsoftmax)); // factors don't
    return state;
  }
};

// class to wrap an IEncoderDecoder and a ILogProbStep that are executed in sequence,
// wrapped again in the IEncoderDecoder interface
// @TODO: seems we are conflating an interface defition with its implementation?
// @TODO: needs a better name. Stepwise is an adjective. Classes are things=nouns. StepwiseWhat?
class Stepwise : public IEncoderDecoder {
protected:
  Ptr<IEncoderDecoder> encdec_;
  Ptr<ILogProbStep> cost_;

public:
  Stepwise(Ptr<IEncoderDecoder> encdec, Ptr<ILogProbStep> cost)
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

  virtual Logits build(Ptr<ExpressionGraph> graph,
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
                                 const std::vector<IndexType>& hypIndices,   // [beamIndex * activeBatchSize + batchIndex]
                                 const Words& words,                         // [beamIndex * activeBatchSize + batchIndex]
                                 const std::vector<IndexType>& batchIndices, // [batchIndex]
                                 int beamSize) override {
    auto nextState = encdec_->step(graph, state, hypIndices, words, batchIndices, beamSize);
    return cost_->apply(nextState);
  }

  virtual Logits build(Ptr<ExpressionGraph> /*graph*/,
                       Ptr<data::CorpusBatch> /*batch*/,
                       bool /*clearGraph*/ = true) override {
    ABORT("Wrong wrapper. Use models::Trainer or models::Scorer");
  }

  virtual Ptr<Options> getOptions() override { return encdec_->getOptions(); };

  virtual void setShortlistGenerator(
      Ptr<const data::ShortlistGenerator> shortlistGenerator) override {
    encdec_->setShortlistGenerator(shortlistGenerator);
  };

  virtual Ptr<data::Shortlist> getShortlist() override {
    return encdec_->getShortlist();
  };

  virtual data::SoftAlignment getAlignment() override { return encdec_->getAlignment(); }
};

}  // namespace models
}  // namespace marian
