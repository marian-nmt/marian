#pragma once

#include <memory>

#include "common/scorer.h"
#include "common/sentence.h"
#include "common/base_best_hyps.h"
#include "common/history.h"


class Search {
  public:
    Search(size_t threadId);
    Histories Decode(const Sentences& sentences);

  private:
    size_t MakeFilter(const Words& srcWords, size_t vocabSize);
    std::vector<ScorerPtr> scorers_;
    Words filterIndices_;
    BestHypsType BestHyps_;
};
