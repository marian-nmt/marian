#pragma once

#include "marian.h"

#include "data/shortlist.h"
#include "models/model_factory.h"

namespace marian {

class ScorerState {
public:
  virtual Expr getLogProbs() = 0;

  virtual float breakDown(size_t i) { return getLogProbs()->val()->get(i); }

  virtual void blacklist(Expr totalCosts, Ptr<data::CorpusBatch> batch){};
};

class Scorer {
protected:
  std::string name_;
  float weight_;

public:
  Scorer(const std::string& name, float weight)
      : name_(name), weight_(weight) {}

  std::string getName() { return name_; }
  float getWeight() { return weight_; }

  virtual void clear(Ptr<ExpressionGraph>) = 0;
  virtual Ptr<ScorerState> startState(Ptr<ExpressionGraph>,
                                      Ptr<data::CorpusBatch>)
      = 0;
  virtual Ptr<ScorerState> step(Ptr<ExpressionGraph>,
                                Ptr<ScorerState>,
                                const std::vector<size_t>&,
                                const std::vector<size_t>&,
                                int dimBatch,
                                int beamSize)
      = 0;

  virtual void init(Ptr<ExpressionGraph> graph) {}

  virtual void setShortlistGenerator(
      Ptr<data::ShortlistGenerator> shortlistGenerator){};
  virtual Ptr<data::Shortlist> getShortlist() { return nullptr; };
  virtual std::vector<float> getAlignment() { return {}; };
};

class ScorerWrapperState : public ScorerState {
protected:
  Ptr<DecoderState> state_;

public:
  ScorerWrapperState(Ptr<DecoderState> state) : state_(state) {}

  virtual Ptr<DecoderState> getState() { return state_; }

  virtual Expr getLogProbs() override { return state_->getLogProbs(); };

  virtual void blacklist(Expr totalCosts, Ptr<data::CorpusBatch> batch) override {
    state_->blacklist(totalCosts, batch);
  }
};

// class to wrap EncoderDecoderBase in a Scorer interface
class ScorerWrapper : public Scorer {
private:
  Ptr<EncoderDecoderBase> encdec_;
  std::string fname_;
  const void* ptr_;

public:
  ScorerWrapper(Ptr<models::ModelBase> encdec,
                const std::string& name,
                float weight,
                const std::string& fname)
      : Scorer(name, weight),
        encdec_(std::static_pointer_cast<EncoderDecoderBase>(encdec)),
        fname_(fname),
        ptr_{0} {}

  ScorerWrapper(Ptr<models::ModelBase> encdec,
                const std::string& name,
                float weight,
                const void* ptr)
      : Scorer(name, weight),
        encdec_(std::static_pointer_cast<EncoderDecoderBase>(encdec)),
        ptr_{ptr} {}

  virtual void init(Ptr<ExpressionGraph> graph) override {
    graph->switchParams(getName());
    if(ptr_)
      encdec_->mmap(graph, ptr_);
    else
      encdec_->load(graph, fname_);
  }

  virtual void clear(Ptr<ExpressionGraph> graph) override {
    graph->switchParams(getName());
    encdec_->clear(graph);
  }

  virtual Ptr<ScorerState> startState(Ptr<ExpressionGraph> graph,
                                      Ptr<data::CorpusBatch> batch) override {
    graph->switchParams(getName());
    return New<ScorerWrapperState>(encdec_->startState(graph, batch));
  }

  virtual Ptr<ScorerState> step(Ptr<ExpressionGraph> graph,
                                Ptr<ScorerState> state,
                                const std::vector<size_t>& hypIndices,
                                const std::vector<size_t>& embIndices,
                                int dimBatch,
                                int beamSize) override {
    graph->switchParams(getName());
    auto wrapperState = std::dynamic_pointer_cast<ScorerWrapperState>(state);
    auto newState = encdec_->step(graph, wrapperState->getState(), hypIndices, embIndices, dimBatch, beamSize);
    return New<ScorerWrapperState>(newState);
  }

  virtual void setShortlistGenerator(
      Ptr<data::ShortlistGenerator> shortlistGenerator) override {
    encdec_->setShortlistGenerator(shortlistGenerator);
  };

  virtual Ptr<data::Shortlist> getShortlist() override {
    return encdec_->getShortlist();
  };

  virtual std::vector<float> getAlignment() override {
    return encdec_->getAlignment().front();
  }
};

Ptr<Scorer> scorerByType(const std::string& fname,
                         float weight,
                         const std::string& model,
                         Ptr<Config> config);

std::vector<Ptr<Scorer>> createScorers(Ptr<Config> options);

Ptr<Scorer> scorerByType(const std::string& fname,
                         float weight,
                         const void* ptr,
                         Ptr<Config> config);

std::vector<Ptr<Scorer>> createScorers(Ptr<Config> options,
                                       const std::vector<const void*>& ptrs);
}  // namespace marian
