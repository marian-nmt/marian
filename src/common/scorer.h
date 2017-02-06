#pragma once

#include <vector>
#include <memory>

#include "common/hypothesis.h"
#include "common/sentence.h"
#include "common/base_matrix.h"
#include "yaml-cpp/node/node.h"

namespace amunmt {

class God;

class State {
  public:
	State() {}
	State(const State &) = delete;
	virtual ~State() {}

    template <class T>
    T& get() {
      return static_cast<T&>(*this);
    }

    template <class T>
    const T& get() const {
      return static_cast<const T&>(*this);;
    }

    virtual std::string Debug() const = 0;

};

typedef std::shared_ptr<State> StatePtr;
typedef std::vector<StatePtr> States;

class Scorer {
  public:
    Scorer(const std::string& name,
           const YAML::Node& config, size_t tab);

    virtual ~Scorer() {}

    virtual void Decode(const God &god, const State& in,
                       State& out, const std::vector<size_t>& beamSizes) = 0;

    virtual void BeginSentenceState(State& state, size_t batchSize=1) = 0;

    virtual void AssembleBeamState(const State& in,
                                   const Beam& beam,
                                   State& out) = 0;

    virtual void SetSource(const Sentences& sources) = 0;

    virtual void Filter(const std::vector<size_t>&) = 0;

    virtual State* NewState() const = 0;

    virtual size_t GetVocabSize() const = 0;

    virtual void CleanUpAfterSentence() {}

    virtual const std::string& GetName() const {
      return name_;
    }

    virtual BaseMatrix& GetProbs() = 0;

  protected:
    const std::string& name_;
    const YAML::Node& config_;
    size_t tab_;
};

class SourceIndependentScorer : public Scorer {
  public:
    SourceIndependentScorer(const std::string& name,
                            const YAML::Node& config, size_t)
    : Scorer(name, config, 0) {}

    virtual ~SourceIndependentScorer() {}

    virtual void SetSource(const Sentences&) {}
};

typedef std::shared_ptr<Scorer> ScorerPtr;

}
