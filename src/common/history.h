#pragma once

#include <queue>

#include "god.h"
#include "hypothesis.h"

namespace amunmt {

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

    History(const God &god, const History &) = delete;

  public:
    History(const God &god, size_t lineNo);

    void Add(const Beam& beam, bool last = false) {
      if (beam.back()->GetPrevHyp() != nullptr) {
        for (size_t j = 0; j < beam.size(); ++j)
          if(beam[j]->GetWord() == EOS || last) {
            float cost = normalize_ ? beam[j]->GetCost() / history_.size() : beam[j]->GetCost();
            topHyps_.push({ history_.size(), j, cost });
          }
      }
      history_.push_back(beam);
    }

    size_t size() const {
      return history_.size();
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

};

///////////////////////////////////////////////////////////////////////////////
//typedef std::vector<History> Histories;
class Histories {
 public:
  Histories() {} // for all histories in translation task
  Histories(const God &god, const Sentences& sentences);

  std::shared_ptr<History> at(size_t id) const {
    return coll_.at(id);
  }

  size_t size() const {
    return coll_.size();
  }

  void SortByLineNum();
  void Append(const Histories &other);

 protected:
  std::vector< std::shared_ptr<History> > coll_;

  Histories(const Histories &) = delete;
};

}
