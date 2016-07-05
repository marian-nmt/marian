#pragma once

#include <memory>
#include <chrono>

#include <unordered_map>
#include <boost/iterator/permutation_iterator.hpp>

#include "common/filter.h"

#include "decoder/god.h"
#include "decoder/sentence.h"
#include "decoder/history.h"
#include "decoder/encoder_decoder.h"
#include "decoder/scorer.h"

class Search {
  private:
    std::vector<ScorerPtr> scorers_;
    std::unordered_map<Word, Word> filterMap_;

  public:
    Search(size_t threadId)
    : scorers_(God::GetScorers(threadId)) {}

    size_t MakeFilter(const Words& srcWords, const size_t vocabSize);

    History Decode(const Sentence& sentence);

    void BestHyps(Beam& bestHyps, const Beam& prevHyps,
                  std::vector<mblas::Matrix>& ProbsEnsemble,
                  const size_t beamSize,
                  History& history);

  private:
    void CleanUp();
};
