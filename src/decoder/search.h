#pragma once

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <limits>
#include <sstream>
#include <queue>
#include <boost/timer/timer.hpp>

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
  
class Search {
  private:
    const Weights& model_;
    Encoder encoder_;
    Decoder decoder_;
    
    mblas::Matrix State_, NextState_, BeamState_;
    mblas::Matrix Embeddings_, NextEmbeddings_;
    mblas::Matrix Probs_;
    mblas::Matrix SourceContext_;

  public:
    Search(const Weights& model)
    : model_(model),
      encoder_(model_),
      decoder_(model_)
    {}
    
    History Decode(const Sentence sourceWords, size_t beamSize = 12) {
      encoder_.GetContext(sourceWords, SourceContext_);
      decoder_.EmptyState(State_, SourceContext_, 1);
      decoder_.EmptyEmbedding(Embeddings_, 1);
      
      History history;
      
      Beam prevHyps;
      prevHyps.emplace_back(0, 0, 0.0);
      
      do {
        decoder_.MakeStep(NextState_, Probs_, State_, Embeddings_, SourceContext_);
        
        Beam hyps;
        BestHyps(hyps, prevHyps, Probs_, beamSize);
        history.Add(hyps, history.size() + 1 == sourceWords.size() * 3);
        
        Beam survivors;
        std::vector<size_t> beamWords;
        std::vector<size_t> beamStateIds;
        for(auto& h : hyps) {
          if(h.GetWord() != EOS) {
            survivors.push_back(h);
            beamWords.push_back(h.GetWord());
            beamStateIds.push_back(h.GetPrevStateIndex());
          }
        }
        beamSize = survivors.size();
        
        if(beamSize == 0)
          break;
        
        decoder_.Lookup(NextEmbeddings_, beamWords);
        mblas::Assemble(BeamState_, NextState_, beamStateIds);
        
        mblas::Swap(Embeddings_, NextEmbeddings_);
        mblas::Swap(State_, BeamState_);
        prevHyps.swap(survivors);
        
      } while(history.size() < sourceWords.size() * 3);
      
      return history;
    }
    
    void BestHyps(Beam& bestHyps, const Beam& prevHyps, mblas::Matrix& Probs, const size_t beamSize) {
      using namespace mblas;
      
      Matrix Costs(Probs.Rows(), 1);
      thrust::host_vector<float> vCosts;
      for(const Hypothesis& h : prevHyps)
        vCosts.push_back(h.GetCost());
      thrust::copy(vCosts.begin(), vCosts.end(), Costs.begin());
      
      BroadcastVecColumn(Log(_1) + _2, Probs, Costs);
      
      thrust::device_vector<unsigned> keys(Probs.size());
      thrust::sequence(keys.begin(), keys.end());
      
      // Here it would be nice to have a partial sort instead of full sort
      thrust::sort_by_key(Probs.begin(), Probs.end(),
                          keys.begin(), thrust::greater<float>());
      
      thrust::host_vector<unsigned> bestKeys(beamSize);
      thrust::copy_n(keys.begin(), beamSize, bestKeys.begin());
      thrust::host_vector<float> bestCosts(beamSize);
      thrust::copy_n(Probs.begin(), beamSize, bestCosts.begin());
      
      for(size_t i = 0; i < beamSize; i++) {
        size_t wordIndex = bestKeys[i] % Probs.Cols();
        size_t hypIndex  = bestKeys[i] / Probs.Cols();
        float  cost = bestCosts[i];
        bestHyps.emplace_back(wordIndex, hypIndex, cost);  
      }
    }
};