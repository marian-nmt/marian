#pragma once

#include "marian.h"

#include "data/shortlist.h"
#include "models/model_factory.h"
#include "3rd_party/mio/mio.hpp"

namespace marian {

class ScorerState {
public:
  virtual ~ScorerState(){}

  virtual Logits getLogProbs() const = 0;

  virtual void blacklist(Expr /*totalCosts*/, Ptr<data::CorpusBatch> /*batch*/){};
};

class Scorer {
protected:
  std::string name_;
  float weight_;

public:
  Scorer(const std::string& name, float weight)
      : name_(name), weight_(weight) {}

  virtual ~Scorer(){}

  std::string getName() { return name_; }
  float getWeight() { return weight_; }

  virtual void clear(Ptr<ExpressionGraph>) = 0;
  virtual Ptr<ScorerState> startState(Ptr<ExpressionGraph>,
                                      Ptr<data::CorpusBatch>)
      = 0;
  virtual Ptr<ScorerState> step(Ptr<ExpressionGraph>,
                                Ptr<ScorerState>,
                                const std::vector<IndexType>&,
                                const Words&,
                                const std::vector<IndexType>& batchIndices,
                                int beamSize)
      = 0;

  virtual void init(Ptr<ExpressionGraph>) {}

  virtual void setShortlistGenerator(Ptr<const data::ShortlistGenerator> /*shortlistGenerator*/){};
  virtual Ptr<data::Shortlist> getShortlist() { return nullptr; };

  virtual std::vector<float> getAlignment() { return {}; };
};

class ScorerWrapperState : public ScorerState {
protected:
  Ptr<DecoderState> state_;

public:
  ScorerWrapperState(Ptr<DecoderState> state) : state_(state) {}
  virtual ~ScorerWrapperState() {}

  virtual Ptr<DecoderState> getState() { return state_; }

  virtual Logits getLogProbs() const override { return state_->getLogProbs(); };

  virtual void blacklist(Expr totalCosts, Ptr<data::CorpusBatch> batch) override {
    state_->blacklist(totalCosts, batch);
  }
};

// class to wrap IEncoderDecoder in a Scorer interface
class ScorerWrapper : public Scorer {
private:
  Ptr<IEncoderDecoder> encdec_;
  std::string fname_;
  std::vector<io::Item> items_;
  const void* ptr_;

public:
  ScorerWrapper(Ptr<models::IModel> encdec,
                const std::string& name,
                float weight,
                std::vector<io::Item>& items)
      : Scorer(name, weight),
        encdec_(std::static_pointer_cast<IEncoderDecoder>(encdec)),
        items_(items),
        ptr_{0} {}

  ScorerWrapper(Ptr<models::IModel> encdec,
                const std::string& name,
                float weight,
                const std::string& fname)
      : Scorer(name, weight),
        encdec_(std::static_pointer_cast<IEncoderDecoder>(encdec)),
        fname_(fname),
        ptr_{0} {}

  ScorerWrapper(Ptr<models::IModel> encdec,
                const std::string& name,
                float weight,
                const void* ptr)
      : Scorer(name, weight),
        encdec_(std::static_pointer_cast<IEncoderDecoder>(encdec)),
        ptr_{ptr} {}

  virtual ~ScorerWrapper() {}

  virtual void init(Ptr<ExpressionGraph> graph) override {
    graph->switchParams(getName());
    if(!items_.empty())
      encdec_->load(graph, items_);
    else if(ptr_)
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
                                const std::vector<IndexType>& hypIndices,
                                const Words& words,
                                const std::vector<IndexType>& batchIndices,
                                int beamSize) override {
    graph->switchParams(getName());
    auto wrapperState = std::dynamic_pointer_cast<ScorerWrapperState>(state);
    auto newState = encdec_->step(graph, wrapperState->getState(), hypIndices, words, batchIndices, beamSize);
    return New<ScorerWrapperState>(newState);
  }

  virtual void setShortlistGenerator(
      Ptr<const data::ShortlistGenerator> shortlistGenerator) override {
    encdec_->setShortlistGenerator(shortlistGenerator);
  };

  virtual Ptr<data::Shortlist> getShortlist() override {
    return encdec_->getShortlist();
  };

  virtual std::vector<float> getAlignment() override {
    // This is called during decoding, where alignments only exist for the last time step. Hence front().
    // This makes as copy. @TODO: It should be OK to return this as a const&.
    return encdec_->getAlignment().front(); // [beam depth * max src length * batch size]
  }
};

Ptr<Scorer> scorerByType(const std::string& fname,
                        float weight,
                        std::vector<io::Item> items,
                        Ptr<Options> options);

Ptr<Scorer> scorerByType(const std::string& fname,
                         float weight,
                         const std::string& model,
                         Ptr<Options> config);


std::vector<Ptr<Scorer>> createScorers(Ptr<Options> options);
std::vector<Ptr<Scorer>> createScorers(Ptr<Options> options, const std::vector<std::vector<io::Item>> models);

Ptr<Scorer> scorerByType(const std::string& fname,
                         float weight,
                         const void* ptr,
                         Ptr<Options> config);

std::vector<Ptr<Scorer>> createScorers(Ptr<Options> options, const std::vector<const void*>& ptrs);
std::vector<Ptr<Scorer>> createScorers(Ptr<Options> options, const std::vector<mio::mmap_source>& mmaps);

}  // namespace marian
