#pragma once

#include <queue>
#include <algorithm>

#include "hypothesis.h"

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
    History(size_t lineNo, bool normalizeScore, size_t maxLength);

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


class Histories {
  public:
    Histories() {} // for all histories in translation task
    Histories(const Sentences& sentences, bool normalizeScore);

    std::shared_ptr<History> at(size_t id) const {
      return coll_.at(id);
    }

    size_t size() const {
      return coll_.size();
    }

    void Add(const Beams& beams) {
      for (size_t i = 0; i < size(); ++i) {
        if (!beams[i].empty()) {
          coll_[i]->Add(beams[i]);
        }
      }
    }

    void SortByLineNum();
    void Append(const Histories &other);

    Beam GetFirstHyps() {
      Beam beam;
      for (auto& history : coll_) {
        beam.emplace_back(history->front()[0]);
      }
      return beam;
    }

  protected:
    std::vector<std::shared_ptr<History>> coll_;
    Histories(const Histories &) = delete;
};

}
