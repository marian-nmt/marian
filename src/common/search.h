#pragma once

#include <memory>

#include "common/scorer.h"
#include "common/sentence.h"
#include "common/history.h"

class Search {
  public:
    Search(size_t threadId);
    History Decode(const Sentence& sentence);

  private:
    size_t MakeFilter(const Words& srcWords, const size_t vocabSize);
    std::vector<ScorerPtr> scorers_;
    Words filterIndices_;
};
