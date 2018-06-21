#pragma once
#include "history.h"

namespace amunmt {

class Histories {
  public:
    Histories() {} // for all histories in translation task
    Histories(const Sentences& sentences, bool normalizeScore, float maxLengthMult);

    std::shared_ptr<History> at(unsigned id) const {
      return coll_.at(id);
    }

    unsigned size() const {
      return coll_.size();
    }

    void Add(const Beams& beams) {
      for (unsigned i = 0; i < size(); ++i) {
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

    void SetActive(bool active);
    void SetActive(unsigned id, bool active);
    unsigned NumActive() const;

  protected:
    std::vector<std::shared_ptr<History>> coll_;
    Histories(const Histories &) = delete;
};


}

