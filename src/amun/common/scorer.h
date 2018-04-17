#pragma once

#include <vector>
#include <memory>

#include "common/hypothesis.h"
#include "common/sentence.h"
#include "common/base_tensor.h"
#include "yaml-cpp/node/node.h"

namespace amunmt {

class God;
class Sentences;
class Hypothesis;
typedef std::shared_ptr<Hypothesis> HypothesisPtr;
typedef std::vector<HypothesisPtr> Beam;


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

    virtual std::string Debug(unsigned verbosity = 1) const = 0;

};

typedef std::shared_ptr<State> StatePtr;
typedef std::vector<StatePtr> States;

class Scorer {
  public:
    Scorer(const God &god,
           const std::string& name,
           const YAML::Node& config, unsigned tab);

    virtual ~Scorer() {}

    virtual void Decode(const State& in, State& out, const std::vector<unsigned>& beamSizes) = 0;

    virtual void BeginSentenceState(State& state, unsigned batchSize = 1) = 0;

    virtual void AssembleBeamState(const State& in, const Beam& beam, State& out) = 0;

    virtual void Encode(const Sentences& sources) = 0;

    virtual void Filter(const std::vector<unsigned>&) = 0;

    virtual State* NewState() const = 0;

    virtual unsigned GetVocabSize() const = 0;

    virtual void CleanAfterTranslation() {}

    virtual const std::string& GetName() const {
      return name_;
    }

    virtual BaseTensor& GetProbs() = 0;
    virtual void *GetNBest() = 0; // hack - need to return matrix<NthOut> but NthOut contain cuda code
    virtual const BaseTensor *GetBias() const = 0;

  protected:
    const God &god_;
    const std::string& name_;
    const YAML::Node& config_;
    unsigned tab_;
};

class SourceIndependentScorer : public Scorer {
  public:
    SourceIndependentScorer(const God &god, const std::string& name,
                            const YAML::Node& config, unsigned)
    : Scorer(god, name, config, 0) {}

    virtual ~SourceIndependentScorer() {}

    virtual void SetSource(const Sentences&) {}
};

typedef std::shared_ptr<Scorer> ScorerPtr;

}
