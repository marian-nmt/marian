#pragma once

#include <queue>
#include <algorithm>

#include "hypothesis.h"
#include "beam.h"

namespace amunmt {

class Sentences;

class History {
  private:
    struct HypothesisCoord {
      bool operator<(const HypothesisCoord& hc) const {
        return cost < hc.cost;
      }

      size_t i;
      size_t j;
      float cost;
    };

    History(const History&) = delete;

  public:
    History(const Sentence &sentence, bool normalizeScore, size_t maxLength);

    void Add(const Beam& beam) {
      if (beam.back()->GetPrevHyp() != nullptr) {
        for (size_t j = 0; j < beam.size(); ++j)
          if(beam[j]->GetWord() == EOS_ID || size() == maxLength_ ) {
            float cost = normalize_ ? beam[j]->GetCost() / history_.size() : beam[j]->GetCost();
            topHyps_.push({ history_.size(), j, cost });
          }
      }
      history_.push_back(beam);
    }

    size_t size() const {
      return history_.size();
    }

    Beam& front() {
      return history_.front();
    }

    NBestList NBest(size_t n) const {
      NBestList nbest;
      auto topHypsCopy = topHyps_;
      while (nbest.size() < n && !topHypsCopy.empty()) {
        auto bestHypCoord = topHypsCopy.top();
        topHypsCopy.pop();

        size_t start = bestHypCoord.i;
        size_t j  = bestHypCoord.j;

        Words targetWords;
        HypothesisPtr bestHyp = history_[start][j];
        while(bestHyp->GetPrevHyp() != nullptr) {
          targetWords.push_back(bestHyp->GetWord());
          bestHyp = bestHyp->GetPrevHyp();
        }

        std::reverse(targetWords.begin(), targetWords.end());
        nbest.emplace_back(targetWords, history_[bestHypCoord.i][bestHypCoord.j]);
      }
      return nbest;
    }

    Result Top() const {
      return NBest(1)[0];
    }

    size_t GetLineNum() const
    { return lineNo_; }

  private:
    std::vector<Beam> history_;
    std::priority_queue<HypothesisCoord> topHyps_;
    bool normalize_;
    size_t lineNo_;
    size_t maxLength_;
};


}
