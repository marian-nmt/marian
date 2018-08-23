#pragma once

#include "data/types.h"
#include "hypothesis.h"

#include <queue>

namespace marian {

class History {
private:
  struct HypothesisCoord {
    bool operator<(const HypothesisCoord& hc) const { return pathScore < hc.pathScore; }

    size_t i;
    size_t j;
    float pathScore;
  };

public:
  History(size_t lineNo, float alpha = 1.f, float wp_ = 0.f);

  float LengthPenalty(size_t length) { return std::pow((float)length, alpha_); }
  float WordPenalty(size_t length) { return wp_ * (float)length; }

  void Add(const Beam& beam, Word trgEosId, bool last = false) {
    if(beam.back()->GetPrevHyp() != nullptr) {
      for(size_t j = 0; j < beam.size(); ++j)
        if(beam[j]->GetWord() == trgEosId || last) {
          float pathScore = (beam[j]->GetPathScore() - WordPenalty(history_.size()))
                       / LengthPenalty(history_.size());
          topHyps_.push({history_.size(), j, pathScore});
          // std::cerr << "Add " << history_.size() << " " << j << " " << pathScore
          // << std::endl;
        }
    }
    history_.push_back(beam);
  }

  size_t size() const { return history_.size(); }

  NBestList NBest(size_t n) const {
    NBestList nbest;
    auto topHypsCopy = topHyps_;
    while(nbest.size() < n && !topHypsCopy.empty()) {
      auto bestHypCoord = topHypsCopy.top();
      topHypsCopy.pop();

      size_t start = bestHypCoord.i;
      size_t j = bestHypCoord.j;
      // float c = bestHypCoord.pathScore;
      // std::cerr << "h: " << start << " " << j << " " << c << std::endl;

      Words targetWords;
      Ptr<Hypothesis> bestHyp = history_[start][j];
      while(bestHyp->GetPrevHyp() != nullptr) {
        targetWords.push_back(bestHyp->GetWord());
        // std::cerr << bestHyp->GetWord() << " " << bestHyp << std::endl;
        bestHyp = bestHyp->GetPrevHyp();
      }

      std::reverse(targetWords.begin(), targetWords.end());
      nbest.emplace_back(targetWords,
                         history_[bestHypCoord.i][bestHypCoord.j],
                         bestHypCoord.pathScore);
    }
    return nbest;
  }

  Result Top() const { return NBest(1)[0]; }

  size_t GetLineNum() const { return lineNo_; }

private:
  std::vector<Beam> history_;
  std::priority_queue<HypothesisCoord> topHyps_;
  size_t lineNo_;
  float alpha_;
  float wp_;
};

typedef std::vector<Ptr<History>> Histories;
}  // namespace marian
