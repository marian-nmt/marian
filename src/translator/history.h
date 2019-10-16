#pragma once

#include "data/types.h"
#include "hypothesis.h"

#include <queue>

namespace marian {

// search grid of one batch entry
class History {
private:
  // one hypothesis of a full sentence (reference into search grid)
  struct SentenceHypothesisCoord {
    bool operator<(const SentenceHypothesisCoord& hc) const { return normalizedPathScore < hc.normalizedPathScore; }

    size_t i; // last time step of this sentence hypothesis
    size_t j; // which beam entry
    float normalizedPathScore; // length-normalized sentence score
  };

  float lengthPenalty(size_t length) { return std::pow((float)length, alpha_); }
  float wordPenalty(size_t length) { return wp_ * (float)length; }
public:
  History(size_t lineNo, float alpha = 1.f, float wp_ = 0.f);

  void add(const Beam& beam, Word trgEosId, bool last = false) {
    if(beam.back()->getPrevHyp() != nullptr) {
      for(size_t j = 0; j < beam.size(); ++j)
        if(beam[j]->getWord() == trgEosId || last) {
          float pathScore =
              (beam[j]->getPathScore() - wordPenalty(history_.size())) / lengthPenalty(history_.size());
          topHyps_.push({history_.size(), j, pathScore});
          // std::cerr << "Add " << history_.size() << " " << j << " " << pathScore
          // << std::endl;
        }
    }
    history_.push_back(beam);
  }

  size_t size() const { return history_.size(); } // number of time steps

  NBestList nBest(size_t n) const {
    NBestList nbest;
    for (auto topHypsCopy = topHyps_; nbest.size() < n && !topHypsCopy.empty(); topHypsCopy.pop()) {
      auto bestHypCoord = topHypsCopy.top();

      const size_t start = bestHypCoord.i; // last time step of this hypothesis
      const size_t j     = bestHypCoord.j; // which beam entry
      Ptr<Hypothesis> bestHyp = history_[start][j];
      // float c = bestHypCoord.normalizedPathScore;
      // std::cerr << "h: " << start << " " << j << " " << c << std::endl;

      // trace back best path
      Words targetWords = bestHyp->tracebackWords();

      // note: bestHyp->getPathScore() is not normalized, while bestHypCoord.normalizedPathScore is
      nbest.emplace_back(targetWords, bestHyp, bestHypCoord.normalizedPathScore);
    }
    return nbest;
  }

  Result top() const { return nBest(1)[0]; }

  size_t getLineNum() const { return lineNo_; }

private:
  std::vector<Beam> history_; // [time step][index into beam] search grid
  std::priority_queue<SentenceHypothesisCoord> topHyps_; // all sentence hypotheses (those that reached eos), sorted by score
  size_t lineNo_;
  float alpha_;
  float wp_;
};

typedef std::vector<Ptr<History>> Histories; // [batchDim]
}  // namespace marian
