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

#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>

#include "types.h"
#include "matrix.h"
#include "dl4mt.h"
#include "hypothesis.h"
#include "history.h"
#include "kenlm.h"
#include "threadpool.h"

class Search {
  private:
    
    struct EncoderDecoder {
      EncoderDecoder(const Weights& model)
      : encoder_(model), decoder_(model)
      {}
      
      Encoder encoder_;
      Decoder decoder_;
      
      mblas::Matrix State_, NextState_, BeamState_;
      mblas::Matrix Embeddings_, NextEmbeddings_, Probs_;
      mblas::Matrix SourceContext_;
    };
    
    typedef std::unique_ptr<EncoderDecoder> EncoderDecoderPtr;
  
    std::vector<EncoderDecoderPtr> encDecs_;
    const std::vector<LM>& lms_;
    bool doBreakdown_;
    size_t device_;
    std::vector<mblas::Matrix> LmProbs_;
  
  public:
    Search(const std::vector<std::unique_ptr<Weights>>& models,
           const std::vector<LM>& lms,
           bool doBreakdown = false)
    : lms_(lms),
      doBreakdown_(doBreakdown),
      device_(models[0]->GetDevice())
    {
      for(auto& m : models)
        encDecs_.emplace_back(new EncoderDecoder(*m));
      LmProbs_.resize(lms.size());
    }
    
    History Decode(const Sentence sourceWords, size_t beamSize = 12) {
      using namespace mblas;
      
      History history;
      
      Hypothesis* bos = new Hypothesis(nullptr, 0, 0, 0.0);
      bos->GetCostBreakdown().resize(encDecs_.size() + lms_.size(), 0.0);
      for(auto& lm : lms_) {
        KenlmState state;
        lm.BeginSentenceState(state);
        bos->AddLMState(state);
      }
      Beam prevHyps = { bos };
      history.Add(prevHyps);
      
      for(auto& encDec : encDecs_) {
        encDec->encoder_.GetContext(sourceWords, encDec->SourceContext_);
        encDec->decoder_.EmptyState(encDec->State_, encDec->SourceContext_, 1);
        encDec->decoder_.EmptyEmbedding(encDec->Embeddings_, 1);
      }
      
      const size_t maxLength = sourceWords.size() * 3;
      do {
        std::vector<Matrix*> Probs;
        for(auto& encDec : encDecs_) {
          encDec->decoder_.MakeStep(encDec->NextState_, encDec->Probs_,
                                    encDec->State_, encDec->Embeddings_,
                                    encDec->SourceContext_);
          Probs.push_back(&encDec->Probs_);
        }
        
        Beam hyps;
        BestHyps(hyps, prevHyps, Probs, beamSize);
        history.Add(hyps, history.size() == maxLength);
        
        Beam survivors;
        std::vector<size_t> beamWords;
        std::vector<size_t> beamStateIds;
        for(auto h : hyps) {
          if(h->GetWord() != EOS) {
            survivors.push_back(h);
            beamWords.push_back(h->GetWord());
            beamStateIds.push_back(h->GetPrevStateIndex());
          }
        }
        beamSize = survivors.size();
        
        if(beamSize == 0)
          break;
        
        for(auto& encDec : encDecs_) {
          encDec->decoder_.Lookup(encDec->NextEmbeddings_, beamWords);
          Assemble(encDec->BeamState_, encDec->NextState_, beamStateIds);
          Swap(encDec->Embeddings_, encDec->NextEmbeddings_);
          Swap(encDec->State_, encDec->BeamState_);
        }
        
        prevHyps.swap(survivors);
        
      } while(history.size() <= maxLength);
      
      return history;
    }
    
    void CalcLMProbs(mblas::Matrix& LmProbs, std::vector<KenlmState>& states,
                     const Beam& prevHyps, const LM& lm) {
      
      size_t rows = LmProbs.Rows();
      size_t cols = LmProbs.Cols();
      
      std::vector<float> costs(rows * cols);
      states.resize(rows * cols);

      {
        ThreadPool pool(4);
        for(size_t i = 0; i < prevHyps.size(); i++) {
          auto call = [i, cols, &prevHyps, &lm, &costs, &states] {
            const KenlmState state = prevHyps[i]->GetLMStates()[lm.GetIndex()];
            KenlmState stateUnk;
            float costUnk = lm.Score(state, 0, stateUnk);
            std::fill(costs.begin() + i * cols, costs.begin() + i * cols + cols, costUnk);
            std::fill(states.begin() + i * cols, states.begin() + i * cols + cols, stateUnk);
            for(auto& wp : lm) {
              costs[i * cols + wp.second] = lm.Score(state, wp.first, states[i * cols + wp.second]);
            }
          };
          pool.enqueue(call);
        }
      }
      cudaSetDevice(device_);
      thrust::copy(costs.begin(), costs.end(), LmProbs.begin());
    }
    
    void BestHyps(Beam& bestHyps, const Beam& prevHyps,
                  std::vector<mblas::Matrix*>& ProbsEnsemble,
                  const size_t beamSize) {
      using namespace mblas;
      
      Matrix& Probs = *ProbsEnsemble[0];
      
      Matrix Costs(Probs.Rows(), 1);
      thrust::host_vector<float> vCosts;
      for(const Hypothesis* h : prevHyps)
        vCosts.push_back(h->GetCost());
      thrust::copy(vCosts.begin(), vCosts.end(), Costs.begin());
      
      BroadcastVecColumn(Log(_1) + _2, Probs, Costs);
      for(size_t i = 1; i < ProbsEnsemble.size(); ++i)
        Element(_1 + Log(_2), Probs, *ProbsEnsemble[i]);
      
      std::vector<std::vector<KenlmState>> states(lms_.size());
      if(!lms_.empty()) {
        for(auto& lm : lms_) {
          size_t index = lm.GetIndex();
          LmProbs_[index].Resize(Probs.Rows(), Probs.Cols());
          CalcLMProbs(LmProbs_[index], states[lm.GetIndex()], prevHyps, lm);
          Element(_1 + lm.GetWeight() * _2, Probs, LmProbs_[index]);
        }
      }
      
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
          auto it = thrust::make_permutation_iterator(ProbsEnsemble[i]->begin(), keys.begin());
          thrust::copy(it, it + beamSize, modelCosts.begin());
          breakDowns.push_back(modelCosts);
        }
        for(size_t i = 0; i < lms_.size(); ++i) {
          thrust::host_vector<float> modelCosts(beamSize);
          auto it = thrust::make_permutation_iterator(LmProbs_[i].begin(), keys.begin());
          thrust::copy(it, it + beamSize, modelCosts.begin());
          breakDowns.push_back(modelCosts);
        }
      }
    
      for(size_t i = 0; i < beamSize; i++) {
        size_t wordIndex = bestKeys[i] % Probs.Cols();
        size_t hypIndex  = bestKeys[i] / Probs.Cols();
        float cost = bestCosts[i];
        Hypothesis* hyp = new Hypothesis(prevHyps[hypIndex], wordIndex, hypIndex, cost);
        for(auto& lm : lms_)
          hyp->AddLMState(states[lm.GetIndex()][bestKeys[i]]);
        
        if(doBreakdown_) {
          float sum = 0;
          for(size_t j = 0; j < ProbsEnsemble.size() + lms_.size(); ++j) {
            if(j == 0)
              hyp->GetCostBreakdown().push_back(breakDowns[j][i]);
            else {
              float cost = 0;
              if(j < ProbsEnsemble.size()) {
                cost = log(breakDowns[j][i]) + const_cast<Hypothesis*>(prevHyps[hypIndex])->GetCostBreakdown()[j];
                sum += cost;  
              }
              else {
                cost = breakDowns[j][i] + const_cast<Hypothesis*>(prevHyps[hypIndex])->GetCostBreakdown()[j];
                sum += lms_[j - ProbsEnsemble.size()].GetWeight() * cost;
              }
              hyp->GetCostBreakdown().push_back(cost);
            }
          }
          hyp->GetCostBreakdown()[0] -= sum;
        }
        bestHyps.push_back(hyp);  
      }
    }
};
