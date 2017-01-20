#pragma once

#include <memory>
#include "common/scorer.h"
#include "common/sentence.h"
#include "common/base_best_hyps.h"
#include "common/history.h"


class Search {
  public:
    Search(God &god, size_t threadId);
    std::shared_ptr<Histories> Decode(God &god, const Sentences& sentences);

  private:
    Search(const Search &) = delete;

    size_t MakeFilter(God &god, const std::set<Word>& srcWords, size_t vocabSize);
    void InitScorers(const Sentences& sentences, States& states, States& nextStates);

    std::vector<ScorerPtr> scorers_;
    Words filterIndices_;
    BestHypsBase &bestHyps_;
};
