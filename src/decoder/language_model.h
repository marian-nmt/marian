#pragma once

#include <vector>

#include "types.h"
#include "scorer.h"
#include "matrix.h"
#include "dl4mt.h"
#include "threadpool.h"
#include "kenlm.h"

class LanguageModelState : public State {
  public:
    std::vector<KenlmState>& GetStates() {
      return states_;
    }
  
    const std::vector<KenlmState>& GetStates() const {
      return states_;
    }
  
  private:
    std::vector<KenlmState> states_;
};

class LanguageModel : public SourceIndependentScorer {
  private:
    typedef LanguageModelState LMState;
    
  public:
    LanguageModel(const LM& lm);
    
    virtual void Score(const State& in,
                       Prob& prob,
                       State& out);
    
    virtual State* NewState();
    
    virtual void BeginSentenceState(State& state);
    
    virtual void AssembleBeamState(const State& in,
                                   const Beam& beam,
                                   State& out);
    
    virtual size_t GetVocabSize() const;

  private:
    const LM& lm_;
};
