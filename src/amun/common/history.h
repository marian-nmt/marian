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

      unsigned i;
      unsigned j;
      float cost;
    };

    History(const History&) = delete;

  public:
    History(const Sentence &sentence, bool normalizeScore, unsigned maxLength);

    void Add(const Beam& beam);

    unsigned size() const {
      return history_.size();
    }

    Beam& front() {
      return history_.front();
    }

    NBestList NBest(unsigned n) const;

    Result Top() const {
      return NBest(1)[0];
    }

    unsigned GetLineNum() const
    { return lineNo_; }

    unsigned GetMaxLength() const
    { return maxLength_; }

    void SetActive(bool active);
    bool GetActive() const;

  private:
    std::vector<Beam> history_;
    std::priority_queue<HypothesisCoord> topHyps_;
    bool normalize_;
    unsigned lineNo_;
    unsigned maxLength_;
    bool active_;
};


}
