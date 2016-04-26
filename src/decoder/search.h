#pragma once

#include <memory>

#include "god.h"
#include "sentence.h"
#include "history.h"

template <class Backend>
class Search {
  private:  
    std::vector<ScorerPtr> scorers_;
  
    using Matrix = typename Backend::Payload;
    
    template <typename T>
    using DeviceVector = typename Backend::DeviceVector<T>;
    
    template <typename T>
    using HostVector = typename Backend::HostVector<T>;
  
  public:
    Search(size_t threadId)
    : scorers_(God::GetScorers(threadId)) {}
    
    History Decode(const Sentence& sentence) {
      boost::timer::cpu_timer timer;
      
      size_t beamSize = God::Get<size_t>("beam-size");
      bool normalize = God::Get<bool>("normalize");
      size_t vocabSize = scorers_[0]->GetVocabSize();
      
      History history;
    
      Beam prevHyps = { HypothesisPtr(new Hypothesis()) };
      history.Add(prevHyps);
      
      States states(scorers_.size());
      States nextStates(scorers_.size());
      std::vector<Matrix> probs(scorers_.size());
      
      for(size_t i = 0; i < scorers_.size(); i++) {
        scorers_[i]->SetSource(sentence);
        
        states[i].reset(scorers_[i]->NewState());
        nextStates[i].reset(scorers_[i]->NewState());
        
        scorers_[i]->BeginSentenceState(*states[i]);
      }
      
      const size_t maxLength = sentence.GetWords().size() * 3;
      do {
        for(size_t i = 0; i < scorers_.size(); i++) {
          (*probs[i]).Resize(beamSize, vocabSize);
          scorers_[i]->Score(*states[i], probs[i], *nextStates[i]);
        }

        // Looking at attention vectors        
        //mblas::Backend::Matrix A;
        //std::static_pointer_cast<EncoderDecoder>(scorers_[0])->GetAttention(A);
        //mblas::debug1(A, 0, sentence.GetWords().size());
        
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
      
      for(auto&& scorer : scorers_)
        scorer->CleanUpAfterSentence();
      return history;
    }

    void BestHyps(Beam& bestHyps, const Beam& prevHyps,
                  std::vector<Matrix>& probsEnsemble,
                  const size_t beamSize) {
      using namespace mblas;
      
      auto& weights = God::GetScorerWeights();
      
      Matrix& probs = probsEnsemble[0];
      
      Matrix costs;
      (*costs).Resize((*probs).Rows(), 1);
      HostVector<float> vCosts;
      for(auto& h : prevHyps)
        vCosts.push_back(h->GetCost());
      Backend::copy(vCosts.begin(), vCosts.end(), costs.begin());
      
      Backend::Broadcast(weights[0] * Backend::_1 + Backend::_2,
                         probs, costs);
      for(size_t i = 0; i < probsEnsemble.size(); ++i)
        Backend::Element(Backend::_1 + weights[i] * Backend::_2,
                         probs, probsEnsemble[i]);
      
      HostVector<unsigned> bestKeys(beamSize);
      HostVector<float> bestCosts(beamSize);
      
      Backend::PartialSortByKey(probs, bestKeys, bestCosts);
      
      std::vector<HostVector<float>> breakDowns;
      bool doBreakdown = God::Get<bool>("n-best");
      if(doBreakdown) {
        breakDowns.push_back(bestCosts);
        for(size_t i = 1; i < probsEnsemble.size(); ++i) {
          HostVector<float> modelCosts(beamSize);
          auto it = Backend::make_permutation_iterator(probsEnsemble[i].begin(), bestKeys.begin());
          Backend::copy(it, it + beamSize, modelCosts.begin());
          breakDowns.push_back(modelCosts);
        }
      }
    
      for(size_t i = 0; i < beamSize; i++) {
        size_t wordIndex = bestKeys[i] % probs.Cols();
        size_t hypIndex  = bestKeys[i] / probs.Cols();
        float cost = bestCosts[i];
        
        HypothesisPtr hyp(new Hypothesis(prevHyps[hypIndex], wordIndex, hypIndex, cost));
        
        if(doBreakdown) {
          hyp->GetCostBreakdown().resize(probsEnsemble.size());
          float sum = 0;
          for(size_t j = 0; j < probsEnsemble.size(); ++j) {
            if(j == 0)
              hyp->GetCostBreakdown()[0] = breakDowns[0][i];
            else {
              float cost = 0;
              if(j < probsEnsemble.size()) {
                if(prevHyps[hypIndex]->GetCostBreakdown().size() < probsEnsemble.size())
                  const_cast<HypothesisPtr&>(prevHyps[hypIndex])->GetCostBreakdown().resize(probsEnsemble.size(), 0.0);
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
