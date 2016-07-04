#pragma once

#include <memory>
#include <chrono>

#include "god.h"
#include "sentence.h"
#include "history.h"
#include "encoder_decoder.h"
#include <boost/iterator/permutation_iterator.hpp>

class Search {
  private:
    std::vector<ScorerPtr> scorers_;
  
  public:
    Search(size_t threadId)
    : scorers_(God::GetScorers(threadId)) {}
    
    void MakeFilter(std::vector<size_t>& filterIds, const Sentence& sentence, size_t vocabSize) {
      for(size_t i = 0; i < 10000; ++i)
        filterIds.push_back(i);
    }
    
    History Decode(const Sentence& sentence);

    void BestHyps(Beam& bestHyps, const Beam& prevHyps,
                  std::vector<mblas::Matrix>& ProbsEnsemble,
                  const size_t beamSize,
                  History& history);

};
