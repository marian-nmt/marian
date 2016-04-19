#pragma once

#include <vector>

#include "types.h"
#include "matrix.h"

typedef mblas::Matrix Prob;
typedef std::vector<Prob> Probs;

class State {
  public:
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
    virtual void Score(const State& in,
                       Prob& prob,
                       State& out) = 0;
    
    virtual void BeginSentenceState(State& state) = 0;
    
    virtual void AssembleBeamState(const State& in,
                                   const Beam& beam,
                                   State& out) = 0;
    
    virtual void SetSource(const Sentence& source) = 0;
    
    virtual State* NewState() = 0;
};

class SourceIndependentScorer : public Scorer {
  public:
    virtual void SetSource(const Sentence& source) {}
};

typedef std::unique_ptr<Scorer> ScorerPtr;
