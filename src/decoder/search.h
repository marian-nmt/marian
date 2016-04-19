#pragma once

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <limits>
#include <sstream>
#include <queue>
#include <set>
#include <boost/timer/timer.hpp>
#include <thread>
#include <algorithm>

#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>

#include "types.h"
#include "matrix.h"
#include "hypothesis.h"
#include "history.h"
#include "threadpool.h"
#include "encoder_decoder.h"
#include "language_model.h"

class Search {
  private:  
    std::vector<ScorerPtr> scorers_;
    const std::vector<float> weights_;
    bool normalize_;
    bool doBreakdown_;
    size_t device_;
    std::vector<mblas::Matrix> LmProbs_;
  
  public:
    Search(const std::vector<std::unique_ptr<Weights>>& models,
           const std::vector<LM>& lms,
           const std::vector<float> weights,
           bool normalize = false,
           bool doBreakdown = false)
    : weights_(weights),
      normalize_(normalize),
      doBreakdown_(doBreakdown),
      device_(models[0]->GetDevice())
    {
      for(auto& m : models)
        scorers_.emplace_back(new EncoderDecoder(*m));
      for(auto& lm : lms)
        scorers_.emplace_back(new LanguageModel(lm));
    }
    
    History Decode(const Sentence sourceWords, size_t beamSize = 12) {
      using namespace mblas;
      
      History history(normalize_);
      
      size_t vocabSize = 85000; // evil, where can I get that from?
                                // max from all vocab sizes?
      
      Hypothesis* bos = new Hypothesis(nullptr, 0, 0, 0.0);
      bos->GetCostBreakdown().resize(scorers_.size(), 0.0);
      Beam prevHyps = { bos };
      history.Add(prevHyps);
      
      States states(scorers_.size());
      States nextStates(scorers_.size());
      Probs probs(scorers_.size());
      
      for(size_t i = 0; i < scorers_.size(); i++) {
        scorers_[i]->SetSource(sourceWords);
        
        states[i].reset(scorers_[i]->NewState());
        nextStates[i].reset(scorers_[i]->NewState());
        
        scorers_[i]->BeginSentenceState(*states[i]);
      }
      
      const size_t maxLength = sourceWords.size() * 3;
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
      
      return history;
    }

    void BestHyps(Beam& bestHyps, const Beam& prevHyps,
                  std::vector<mblas::Matrix>& ProbsEnsemble,
                  const size_t beamSize) {
      using namespace mblas;
      
      Matrix& Probs = ProbsEnsemble[0];
      
      Matrix Costs(Probs.Rows(), 1);
      thrust::host_vector<float> vCosts;
      for(const Hypothesis* h : prevHyps)
        vCosts.push_back(h->GetCost());
      thrust::copy(vCosts.begin(), vCosts.end(), Costs.begin());
      
      BroadcastVecColumn(weights_[0] * _1 + _2, Probs, Costs);
      for(size_t i = 1; i < ProbsEnsemble.size(); ++i)
        Element(_1 + weights_[i] * _2, Probs, ProbsEnsemble[i]);
      
      thrust::device_vector<unsigned> keys(Probs.size());
      thrust::host_vector<unsigned> bestKeys(beamSize);
      thrust::host_vector<float> bestCosts(beamSize);
      
      // @TODO: Here we need to have a partial sort
      if(beamSize < 10) {
        for(size_t i = 0; i < beamSize; ++i) {
          thrust::device_vector<float>::iterator iter =
            thrust::max_element(Probs.begin(), Probs.end());
          bestKeys[i] = iter - Probs.begin();
          bestCosts[i] = *iter;
          *iter = std::numeric_limits<float>::lowest();
        }
        thrust::copy(bestKeys.begin(), bestKeys.end(), keys.begin());
      }
      else {
        thrust::sequence(keys.begin(), keys.end());
        thrust::sort_by_key(Probs.begin(), Probs.end(),
                            keys.begin(), thrust::greater<float>());
      
        thrust::copy_n(keys.begin(), beamSize, bestKeys.begin());
        thrust::copy_n(Probs.begin(), beamSize, bestCosts.begin());
      }
      
      std::vector<thrust::host_vector<float>> breakDowns;
      if(doBreakdown_) {
        breakDowns.push_back(bestCosts);
        for(size_t i = 1; i < ProbsEnsemble.size(); ++i) {
          thrust::host_vector<float> modelCosts(beamSize);
          auto it = thrust::make_permutation_iterator(ProbsEnsemble[i].begin(), keys.begin());
          thrust::copy(it, it + beamSize, modelCosts.begin());
          breakDowns.push_back(modelCosts);
        }
      }
    
      for(size_t i = 0; i < beamSize; i++) {
        size_t wordIndex = bestKeys[i] % Probs.Cols();
        size_t hypIndex  = bestKeys[i] / Probs.Cols();
        float cost = bestCosts[i];
        Hypothesis* hyp = new Hypothesis(prevHyps[hypIndex], wordIndex, hypIndex, cost);
        
        if(doBreakdown_) {
          float sum = 0;
          for(size_t j = 0; j < ProbsEnsemble.size(); ++j) {
            if(j == 0)
              hyp->GetCostBreakdown().push_back(breakDowns[j][i]);
            else {
              float cost = 0;
              if(j < ProbsEnsemble.size())
                cost = breakDowns[j][i] + const_cast<Hypothesis*>(prevHyps[hypIndex])->GetCostBreakdown()[j];
              sum += weights_[j] * cost;  
              hyp->GetCostBreakdown().push_back(cost);
            }
          }
          hyp->GetCostBreakdown()[0] -= sum;
          hyp->GetCostBreakdown()[0] /= weights_[0];
        }
        bestHyps.push_back(hyp);  
      }
    }
};
