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

  public:
    History(size_t lineNo, bool normalize);

    void Add(const Beam& beam, bool last = false);

    NBestList NBest(size_t n) const;
    Result Top() const;


    size_t GetLineNum() const;

    size_t size() const;

  private:
    std::vector<Beam> history_;
    std::priority_queue<HypothesisCoord> topHyps_;
    bool normalize_;
    size_t lineNo_;

  private:
    History(const History &) = delete;

};

///////////////////////////////////////////////////////////////////////////////

class Histories {
  public:
    Histories();
    Histories(const Sentences& sentences, bool normalize=true);

    std::shared_ptr<History> at(size_t id) const;

    size_t size() const;

    void SortByLineNum();
    void Append(const Histories &other);

  protected:
    std::vector<std::shared_ptr<History>> coll_;

  protected:
    Histories(const Histories&) = delete;
};

}
