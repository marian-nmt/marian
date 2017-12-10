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

    void Add(const Beam& beam);

    size_t size() const {
      return history_.size();
    }

    Beam& front() {
      return history_.front();
    }

    NBestList NBest(size_t n) const;

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
