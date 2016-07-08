#pragma once

#include "decoder/scorer.h"

class Sentence;
class History;

class Search {
  private:
    std::vector<ScorerPtr> scorers_;
    Words filterIndices_;

  public:
    Search(size_t threadId);

    size_t MakeFilter(const Words& srcWords, const size_t vocabSize);

    History Decode(const Sentence& sentence);

    void BestHyps(Beam& bestHyps,
                  const Beam& prevHyps,
                  std::vector<mblas::Matrix>& ProbsEnsemble,
                  const size_t beamSize,
                  History& history);

  private:
    void CleanUp();
};
