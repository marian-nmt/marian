#pragma once

#include <vector>
#include <memory>

#include "hypothesis.h"
#include "sentence.h"
#include "base_matrix.h"
#include "yaml-cpp/node/node.h"

class State {
  public:
    virtual ~State() {}
    
    template <class T>
    T& get() {
      return static_cast<T&>(*this);
    }
    
    template <class T>
    const T& get() const {
      return static_cast<const T&>(*this);;
    }
};

typedef std::unique_ptr<State> StatePtr;
typedef std::vector<StatePtr> States;

class Scorer {
  public:
    Scorer(const std::string& name,
           const YAML::Node& config, size_t tab);
    
    virtual ~Scorer() {}
    
    virtual void Score(const State& in,
                       mblas::BaseMatrix& prob,
                       State& out) = 0;
    
    virtual void BeginSentenceState(State& state) = 0;
    
    virtual void AssembleBeamState(const State& in,
                                   const Beam& beam,
                                   State& out) = 0;
    
    virtual void SetSource(const Sentence& source) = 0;
    
    virtual State* NewState() = 0;
    
    virtual size_t GetVocabSize() const = 0;
    
    virtual void CleanUpAfterSentence() {}
    
    virtual const std::string& GetName() const {
      return name_;
    }
    
    virtual mblas::BaseMatrix *CreateMatrix() = 0;

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
    
    virtual void SetSource(const Sentence& source) {}
};

typedef std::shared_ptr<Scorer> ScorerPtr;
