#pragma once

#include <vector>

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
    LanguageModel(const LM& lm)
    : lm_(lm)
    {}
    
    virtual void Score(const State& in,
                       Prob& prob,
                       State& out) {
      
      const LMState& lmIn = in.get<LMState>();
      LMState& lmOut = out.get<LMState>();
      
      size_t rows = prob.Rows();
      size_t cols = prob.Cols();
      
      std::vector<float> costs(rows * cols);
      std::vector<KenlmState>& states = lmOut.GetStates();
      states.resize(rows * cols);
    
      for(size_t i = 0; i < lmIn.GetStates().size(); i++) {
          const KenlmState& state = lmIn.GetStates()[i];
          KenlmState stateUnk;
          float costUnk = lm_.Score(state, 0, stateUnk);
          std::fill(costs.begin() + i * cols, costs.begin() + i * cols + cols, costUnk);
          std::fill(states.begin() + i * cols, states.begin() + i * cols + cols, stateUnk);
      }
      
      {  
        ThreadPool pool(8); // this should be a parameter somewhere
        size_t batchSize = 1000; // this, too
        for(size_t batchStart = 0; batchStart < lm_.size(); batchStart += batchSize) {
          auto call = [batchStart, batchSize, cols, &lmIn, this, &costs, &states] {
            size_t batchEnd = min(batchStart + batchSize, lm_.size());
            for(auto it = lm_.begin() + batchStart; it != lm_.begin() + batchEnd; ++it) 
              for(size_t i = 0; i < lmIn.GetStates().size(); i++) {
                const KenlmState state = lmIn.GetStates()[i];
                costs[i * cols + it->second] = lm_.Score(state, it->first, states[i * cols + it->second]);
              }
          };
          pool.enqueue(call);
        }
      }
      
      thrust::copy(costs.begin(), costs.end(), prob.begin());
    }
    
    virtual State* NewState() {
      return new LMState(); 
    }
    
    virtual void BeginSentenceState(State& state) {
      LMState& lmState = state.get<LMState>();
      lmState.GetStates().resize(1);
      lmState.GetStates()[0] = lm_.BeginSentenceState();
    }
    
    virtual void AssembleBeamState(const State& in,
                                   const Beam& beam,
                                   State& out) {
      const LMState& lmIn = in.get<LMState>();
      LMState& lmOut = out.get<LMState>();
      
      lmOut.GetStates().resize(beam.size());
      for(size_t i = 0; i < beam.size(); ++i)
         lmOut.GetStates()[i] = lmIn.GetStates()[beam[i]->GetPrevStateIndex()];
    }
    
  private:
    const LM& lm_;
};
