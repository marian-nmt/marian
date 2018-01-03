#pragma once

#include <vector>
#include <memory>

#include "common/sentence.h"
#include "common/base_matrix.h"
#include "yaml-cpp/node/node.h"

namespace amunmt {

class Histories;
class BestHypsBase;
class Search;

class Hypothesis;
using HypothesisPtr = std::shared_ptr<Hypothesis>;
using Hypotheses = std::vector<HypothesisPtr>;

class Sentences;
using SentencesPtr = std::shared_ptr<Sentences>;

class EncOut;
using EncOutPtr = std::shared_ptr<EncOut>;

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
           const YAML::Node& config, unsigned tab,
           const Search &search);

    virtual ~Scorer();

    virtual void Encode(const SentencesPtr &sources) = 0;

    virtual void Filter(const std::vector<unsigned>&) = 0;

    virtual State* NewState() const = 0;

    virtual unsigned GetVocabSize() const = 0;

    virtual void CleanAfterTranslation() {}

    virtual const std::string& GetName() const {
      return name_;
    }

    virtual BaseMatrix& GetProbs() = 0;
    virtual void *GetNBest() = 0; // hack - need to return matrix<NthOut> but NthOut contain cuda code
    virtual const BaseMatrix *GetBias() const = 0;

  protected:
    const God &god_;
    const Search &search_;
    const std::string& name_;
    const YAML::Node& config_;
    unsigned tab_;
};

class SourceIndependentScorer : public Scorer {
  public:
    SourceIndependentScorer(const God &god, const std::string& name,
                            const YAML::Node& config, unsigned,
                            const Search &search)
    : Scorer(god, name, config, 0, search) {}

    virtual ~SourceIndependentScorer() {}

    virtual void SetSource(const Sentences&) {}
};

typedef std::shared_ptr<Scorer> ScorerPtr;

}
