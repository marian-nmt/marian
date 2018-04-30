#pragma once

#include "models/encoder_decoder.h"
#include "layers/generic.h"
#include "layers/guided_alignment.h"

namespace marian {
namespace models {

class CostBase {
public:
  virtual Expr apply(Ptr<ModelBase> model,
                     Ptr<ExpressionGraph> graph,
                     Ptr<data::Batch> batch,
                     bool clearGraph = true) = 0;
};


class EncoderDecoderCE : public CostBase {
protected:
  Ptr<Options> options_;

public:
  EncoderDecoderCE(Ptr<Options> options)
  : options_(options) {}

  Expr apply(Ptr<ModelBase> model,
             Ptr<ExpressionGraph> graph,
             Ptr<data::Batch> batch,
             bool clearGraph = true) {

    auto encdec = std::static_pointer_cast<EncoderDecoder>(model);
    auto corpusBatch = std::static_pointer_cast<data::CorpusBatch>(batch);

    auto state = encdec->stepAll(graph, corpusBatch, clearGraph);

    std::string costType = options_->get<std::string>("cost-type");
    bool inference = options_->get<bool>("inference", false);

    float ls = inference ? 0.f : options_->get<float>("label-smoothing");

    Expr weights;
    bool sentenceWeighting = false;

    if(options_->has("data-weighting") && !inference) {
      ABORT_IF(corpusBatch->getDataWeights().empty(),
               "Vector of weights is unexpectedly empty!");

      sentenceWeighting
          = options_->get<std::string>("data-weighting-type") == "sentence";
      int dimBatch = corpusBatch->size();
      int dimWords = sentenceWeighting ? 1 : corpusBatch->back()->batchWidth();

      weights = graph->constant({1, dimWords, dimBatch, 1},
                                inits::from_vector(corpusBatch->getDataWeights()));
    }

    auto cost
        = Cost(state->getProbs(),
               state->getTargetIndices(),
               state->getTargetMask(),
               costType,
               ls,
               weights);

    if(options_->has("guided-alignment") && !inference) {
      auto alignments = encdec->getDecoders()[0]->getAlignments();
      ABORT_IF(alignments.empty(), "Model does not seem to support alignments");

      auto att = concatenate(alignments, keywords::axis = 3);

      return cost + guidedAlignmentCost(graph, corpusBatch, options_, att);
    } else {
      return cost;
    }

    return cost;
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
                    bool markedReloaded = true) {
    model_->load(graph, name, markedReloaded);
  };

  virtual void save(Ptr<ExpressionGraph> graph,
                    const std::string& name,
                    bool saveTranslatorConfig = false) {
    model_->save(graph, name, saveTranslatorConfig);
  }

  virtual Expr build(Ptr<ExpressionGraph> graph,
                     Ptr<data::Batch> batch,
                     bool clearGraph = true) {
    return cost_->apply(model_,
                        graph,
                        batch,
                        clearGraph);
  };

  virtual void clear(Ptr<ExpressionGraph> graph) {
    model_->clear(graph);
  };

};

typedef Trainer Scorer;

class CostStep {
public:
  virtual Ptr<DecoderState> apply(Ptr<DecoderState> state) = 0;
};

class LogsoftmaxStep : public CostStep {
public:
  virtual Ptr<DecoderState> apply(Ptr<DecoderState> state) {
    state->setProbs(logsoftmax(state->getProbs()));
    return state;
  }
};

class Stepwise : public EncoderDecoderBase {
protected:
  Ptr<EncoderDecoderBase> encdec_;
  Ptr<CostStep> cost_;

public:
  Stepwise(Ptr<EncoderDecoderBase> encdec, Ptr<CostStep> cost)
  : encdec_(encdec), cost_(cost) {}

  virtual void load(Ptr<ExpressionGraph> graph,
                    const std::string& name,
                    bool markedReloaded = true)  {
    encdec_->load(graph, name, markedReloaded);
  }

  virtual void save(Ptr<ExpressionGraph> graph,
                    const std::string& name,
                    bool saveTranslatorConfig = false) {
    encdec_->save(graph, name, saveTranslatorConfig);
  }

  virtual void clear(Ptr<ExpressionGraph> graph) {
    encdec_->clear(graph);
  }

  virtual Expr build(Ptr<ExpressionGraph> graph,
                     Ptr<data::Batch> batch,
                     bool clearGraph = true) {
    auto corpusBatch = std::static_pointer_cast<data::CorpusBatch>(batch);
    return build(graph, corpusBatch, clearGraph);
  }

  virtual Ptr<DecoderState> startState(Ptr<ExpressionGraph> graph,
                                       Ptr<data::CorpusBatch> batch) {
    return encdec_->startState(graph, batch);
  }

  virtual Ptr<DecoderState> step(Ptr<ExpressionGraph> graph,
                                 Ptr<DecoderState> state,
                                 const std::vector<size_t>& hypIndices,
                                 const std::vector<size_t>& embIndices,
                                 int dimBatch,
                                 int beamSize) {
    auto nextState = encdec_->step(graph, state, hypIndices, embIndices, dimBatch, beamSize);
    return cost_->apply(nextState);
  }

  virtual Expr build(Ptr<ExpressionGraph> graph,
                     Ptr<data::CorpusBatch> batch,
                     bool clearGraph = true) {
    ABORT("Wrong wrapper. Use models::Trainer or models::Scorer");
    return nullptr;
  }

  virtual Ptr<Options> getOptions() {
    return encdec_->getOptions();
  };

  virtual void setShortlistGenerator(Ptr<data::ShortlistGenerator> shortlistGenerator) {
    encdec_->setShortlistGenerator(shortlistGenerator);
  };

  virtual Ptr<data::Shortlist> getShortlist() {
    return encdec_->getShortlist();
  };

};

static Ptr<ModelBase> add_cost(Ptr<EncoderDecoder> encdec, Ptr<Options> options) {
  switch (options->get<usage>("usage", usage::raw)) {
    case usage::training:
      return New<Trainer>(encdec, New<EncoderDecoderCE>(options));
    case usage::scoring:
      return New<Scorer>(encdec, New<EncoderDecoderCE>(options));
    case usage::translation:
      return New<Stepwise>(encdec, New<LogsoftmaxStep>());
    case usage::raw:
    default:
      return encdec;
  }
}

}
}
