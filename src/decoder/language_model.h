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
    LanguageModel(const LM& lm)
    : SourceIndependentScorer(), lm_(lm)
    {}
    
    virtual void Score(const State& in,
                       Prob& prob,
                       State& out) {
      
      const LMState& lmIn = in.get<LMState>();
      LMState& lmOut = out.get<LMState>();
      
      size_t rows = prob.Rows();
      size_t cols = prob.Cols();
      
      std::vector<float> costs(rows * cols);
      const std::vector<KenlmState>& inStates = lmIn.GetStates();
      std::vector<KenlmState>& outStates = lmOut.GetStates();
      outStates.resize(rows * cols);
    
      for(size_t i = 0; i < inStates.size(); i++) {
          KenlmState stateUnk;
          float costUnk = lm_.Score(inStates[i], 0, stateUnk);
          std::fill(costs.begin() + i * cols, costs.begin() + i * cols + cols, costUnk);
          std::fill(outStates.begin() + i * cols, outStates.begin() + i * cols + cols, stateUnk);
      }
      
      {  
        ThreadPool pool(God::Get<size_t>("kenlm-batch-threads")); 
        size_t batchSize = God::Get<size_t>("kenlm-batch-size"); 
        for(size_t batchStart = 0; batchStart < lm_.size(); batchStart += batchSize) {
          auto call = [batchStart, batchSize, cols, this, &costs, &inStates, &outStates] {
            size_t batchEnd = min(batchStart + batchSize, lm_.size());
            for(auto it = lm_.begin() + batchStart; it != lm_.begin() + batchEnd; ++it)
              if(it->second < cols)
                for(size_t i = 0; i < inStates.size(); i++)
                  costs[i * cols + it->second] = lm_.Score(inStates[i], it->first, outStates[i * cols + it->second]);
          };
          pool.enqueue(call);
        }
      }
      algo::copy(costs.begin(), costs.end(), prob.begin());
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
      
      size_t cols = lmIn.GetStates().size() / beam.size();
      
      lmOut.GetStates().resize(beam.size());
      for(size_t i = 0; i < beam.size(); ++i)
         lmOut.GetStates()[i] = lmIn.GetStates()[i * cols + beam[i]->GetWord()];
    }
    
    virtual size_t GetVocabSize() const {
      return lm_.size();
    }
    
  private:
    const LM& lm_;
};
