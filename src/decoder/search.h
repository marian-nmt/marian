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
    
  public:
    Search(const std::vector<std::unique_ptr<Weights>>& models) {
      for(auto& m : models)
        encDecs_.emplace_back(new EncoderDecoder(*m));
    }
    
    History Decode(const Sentence sourceWords, size_t beamSize = 12) {
      using namespace mblas;
      
      History history;
      Beam prevHyps;
      prevHyps.emplace_back(0, 0, 0.0);
      
      for(auto& encDec : encDecs_) {
        encDec->encoder_.GetContext(sourceWords, encDec->SourceContext_);
        encDec->decoder_.EmptyState(encDec->State_, encDec->SourceContext_, 1);
        encDec->decoder_.EmptyEmbedding(encDec->Embeddings_, 1);
      }
      
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
        
        for(auto& encDec : encDecs_) {
          encDec->decoder_.Lookup(encDec->NextEmbeddings_, beamWords);
          Assemble(encDec->BeamState_, encDec->NextState_, beamStateIds);
          Swap(encDec->Embeddings_, encDec->NextEmbeddings_);
          Swap(encDec->State_, encDec->BeamState_);
        }
        
        prevHyps.swap(survivors);
        
      } while(history.size() < sourceWords.size() * 3);
      
      return history;
    }
    
    void BestHyps(Beam& bestHyps, const Beam& prevHyps, std::vector<mblas::Matrix*>& ProbsEnsemble, const size_t beamSize) {
      using namespace mblas;
      
      Matrix& Probs = *ProbsEnsemble[0];
      
      Matrix Costs(Probs.Rows(), 1);
      thrust::host_vector<float> vCosts;
      for(const Hypothesis& h : prevHyps)
        vCosts.push_back(h.GetCost());
      thrust::copy(vCosts.begin(), vCosts.end(), Costs.begin());
      
      BroadcastVecColumn(Log(_1) + _2, Probs, Costs);
      for(size_t i = 1; i < ProbsEnsemble.size(); ++i)
        Element(_1 + Log(_2), Probs, *ProbsEnsemble[i]);
    
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
        //if(costBreakDown_) {
        //  
        //}
        bestHyps.emplace_back(wordIndex, hypIndex, cost);  
      }
    }
};
