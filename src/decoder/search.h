#pragma once

#include "god.h"
#include "sentence.h"
#include "history.h"

class Search {
  private:
    std::vector<ScorerPtr> scorers_;
  
  public:
    Search(size_t threadId)
    : scorers_(God::GetScorers(threadId)) {}
    
    History Decode(const Sentence& sentence) {
      boost::timer::cpu_timer timer;
      
      size_t beamSize = God::Get<size_t>("beam-size");
      bool normalize = God::Get<bool>("normalize");
      
      // @TODO Future: in order to do batch sentence decoding
      // it should be enough to keep track of hypotheses in
      // separate History objects.
      
      History history;
      
      size_t vocabSize = 85000; // evil, where can I get that from?
                                // max from all vocab sizes?
      
      Beam prevHyps = { HypothesisPtr(new Hypothesis()) };
      history.Add(prevHyps);
      
      States states(scorers_.size());
      States nextStates(scorers_.size());
      Probs probs(scorers_.size());
      
      for(size_t i = 0; i < scorers_.size(); i++) {
        scorers_[i]->SetSource(sentence.GetWords());
        
        states[i].reset(scorers_[i]->NewState());
        nextStates[i].reset(scorers_[i]->NewState());
        
        scorers_[i]->BeginSentenceState(*states[i]);
      }
      
      const size_t maxLength = sentence.GetWords().size() * 3;
      do {
        for(size_t i = 0; i < scorers_.size(); i++) {
          probs[i].Resize(beamSize, vocabSize);
          scorers_[i]->Score(*states[i], probs[i], *nextStates[i]);
        }
        
        Beam hyps;
        BestHyps(hyps, prevHyps, probs, beamSize);
        history.Add(hyps, history.size() == maxLength);
        
        Beam survivors;
        for(auto h : hyps)
          if(h->GetWord() != EOS)
            survivors.push_back(h);
        beamSize = survivors.size();
        if(beamSize == 0)
          break;
        
        for(size_t i = 0; i < scorers_.size(); i++)
          scorers_[i]->AssembleBeamState(*nextStates[i], survivors, *states[i]);
        
        prevHyps.swap(survivors);
        
      } while(history.size() <= maxLength);
      
      LOG(progress) << "Line " << sentence.GetLine()
        << ": Search took " << timer.format(3, "%ws");
      
      return history;
    }

    void BestHyps(Beam& bestHyps, const Beam& prevHyps,
                  std::vector<mblas::Matrix>& ProbsEnsemble,
                  const size_t beamSize) {
      using namespace mblas;
      
      auto& weights = God::GetScorerWeights();
      
      Matrix& Probs = ProbsEnsemble[0];
      
      Matrix Costs(Probs.Rows(), 1);
      HostVector<float> vCosts;
      for(auto& h : prevHyps)
        vCosts.push_back(h->GetCost());
      algo::copy(vCosts.begin(), vCosts.end(), Costs.begin());
      
      BroadcastVecColumn(weights[0] * _1 + _2, Probs, Costs);
      for(size_t i = 1; i < ProbsEnsemble.size(); ++i)
        Element(_1 + weights[i] * _2, Probs, ProbsEnsemble[i]);
      
      DeviceVector<unsigned> keys(Probs.size());
      HostVector<unsigned> bestKeys(beamSize);
      HostVector<float> bestCosts(beamSize);
      
      // @TODO: Here we need to have a partial sort
      if(beamSize < 10) {
        for(size_t i = 0; i < beamSize; ++i) {
          DeviceVector<float>::iterator iter =
            algo::max_element(Probs.begin(), Probs.end());
          bestKeys[i] = iter - Probs.begin();
          bestCosts[i] = *iter;
          *iter = std::numeric_limits<float>::lowest();
        }
        algo::copy(bestKeys.begin(), bestKeys.end(), keys.begin());
      }
      else {
        // these two function do not have equivalents in
        // in the standard library or boost, keeping thrust
        // namespace for now
        thrust::sequence(keys.begin(), keys.end());
        thrust::sort_by_key(Probs.begin(), Probs.end(),
                            keys.begin(), algo::greater<float>());
      
        algo::copy_n(keys.begin(), beamSize, bestKeys.begin());
        algo::copy_n(Probs.begin(), beamSize, bestCosts.begin());
      }
      
      std::vector<HostVector<float>> breakDowns;
      bool doBreakdown = God::Get<bool>("n-best");
      if(doBreakdown) {
        breakDowns.push_back(bestCosts);
        for(size_t i = 1; i < ProbsEnsemble.size(); ++i) {
          HostVector<float> modelCosts(beamSize);
          auto it = iteralgo::make_permutation_iterator(ProbsEnsemble[i].begin(), keys.begin());
          algo::copy(it, it + beamSize, modelCosts.begin());
          breakDowns.push_back(modelCosts);
        }
      }
    
      for(size_t i = 0; i < beamSize; i++) {
        size_t wordIndex = bestKeys[i] % Probs.Cols();
        size_t hypIndex  = bestKeys[i] / Probs.Cols();
        float cost = bestCosts[i];
        
        HypothesisPtr hyp(new Hypothesis(prevHyps[hypIndex], wordIndex, hypIndex, cost));
        
        if(doBreakdown) {
          hyp->GetCostBreakdown().resize(ProbsEnsemble.size());
          float sum = 0;
          for(size_t j = 0; j < ProbsEnsemble.size(); ++j) {
            if(j == 0)
              hyp->GetCostBreakdown()[0] = breakDowns[0][i];
            else {
              float cost = 0;
              if(j < ProbsEnsemble.size()) {
                if(prevHyps[hypIndex]->GetCostBreakdown().size() < ProbsEnsemble.size())
                  const_cast<HypothesisPtr&>(prevHyps[hypIndex])->GetCostBreakdown().resize(ProbsEnsemble.size(), 0.0);
                cost = breakDowns[j][i] + const_cast<HypothesisPtr&>(prevHyps[hypIndex])->GetCostBreakdown()[j];
              }
              sum += weights[j] * cost;  
              hyp->GetCostBreakdown()[j] = cost;
            }
          }
          hyp->GetCostBreakdown()[0] -= sum;
          hyp->GetCostBreakdown()[0] /= weights[0];
        }
        bestHyps.push_back(hyp);  
      }
    }
};
