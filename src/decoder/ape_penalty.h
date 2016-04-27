#pragma once

#include <vector>

#include "types.h"
#include "scorer.h"
#include "matrix.h"

class ApePenaltyState : public State {
  // Dummy
};

class ApePenalty : public Scorer {
    
  public:
    ApePenalty(size_t sourceIndex)
    : Scorer(sourceIndex)
    { }
    
    virtual void SetSource(const Sentence& source) {
        const Words& words = source.GetWords(sourceIndex_);
        const Vocab& svcb = God::GetSourceVocab(sourceIndex_);
        const Vocab& tvcb = God::GetTargetVocab();
        
        costs_.clear();
        costs_.resize(tvcb.size(), -1.0);
        for(auto& s : words) {
          const std::string& sstr = svcb[s];
          Word t = tvcb[sstr];
          if(t != UNK && t < costs_.size())
            costs_[t] = 0.0;
        }
    }
    
    virtual void Score(const State& in,
                       Prob& prob,
                       State& out) {
      size_t cols = prob.Cols();
      for(size_t i = 0; i < prob.Rows(); ++i)
        algo::copy(costs_.begin(), costs_.begin() + cols, prob.begin() + i * cols);
    }
    
    virtual State* NewState() {
      return new ApePenaltyState(); 
    }
    
    virtual void BeginSentenceState(State& state) { }
    
    virtual void AssembleBeamState(const State& in,
                                   const Beam& beam,
                                   State& out) { }
    
    virtual size_t GetVocabSize() const {
      return 0;
    }
    
  private:
    std::vector<float> costs_;
};
