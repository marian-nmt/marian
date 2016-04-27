#pragma once

#include <vector>

#include "hypothesis.h"
#include "sentence.h"
#include "matrix.h"

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
typedef mblas::Matrix Prob;
typedef std::vector<Prob> Probs;

class Scorer {
  public:
    Scorer() : sourceIndex_(0) {} 
    Scorer(size_t sourceIndex) : sourceIndex_(sourceIndex) {} 
    
    virtual ~Scorer() {}
    
    virtual void Score(const State& in,
                       Prob& prob,
                       State& out) = 0;
    
    virtual void BeginSentenceState(State& state) = 0;
    
    virtual void AssembleBeamState(const State& in,
                                   const Beam& beam,
                                   State& out) = 0;
    
    virtual void SetSource(const Sentence& source) = 0;
    
    virtual State* NewState() = 0;
    
    virtual size_t GetVocabSize() const = 0;
    
    virtual void CleanUpAfterSentence() {}
    
  protected:
    size_t sourceIndex_;
};

class SourceIndependentScorer : public Scorer {
  public:
    SourceIndependentScorer() : Scorer(0) {} 
    SourceIndependentScorer(size_t) : Scorer(0) {} 
    
    virtual ~SourceIndependentScorer() {}
    
    virtual void SetSource(const Sentence& source) {}
};

typedef std::shared_ptr<Scorer> ScorerPtr;
