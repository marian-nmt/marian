#pragma once

#include <queue>
#include <algorithm>

#include "hypothesis.h"

namespace amunmt {

std::string GetAlignmentString(const std::vector<size_t>& alignment);
std::string GetSoftAlignmentString(const HypothesisPtr& hypothesis);
std::string GetNematusAlignmentString(const HypothesisPtr& hypothesis, std::string best, std::string source, size_t linenum);

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

    void Add(const Hypotheses& beam);

    size_t size() const {
      return history_.size();
    }

    Hypotheses& front() {
      return history_.front();
    }

    NBestList NBest(size_t n) const;

    Result Top() const {
      return NBest(1)[0];
    }

    size_t GetLineNum() const
    { return lineNo_; }

    void Output(const God &god) const;

    void Output(const God &god, std::ostream& out) const;

  private:
    std::vector<Hypotheses> history_;
    std::priority_queue<HypothesisCoord> topHyps_;
    bool normalize_;
    size_t lineNo_;
    size_t maxLength_;
};


}
