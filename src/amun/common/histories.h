#pragma once
#include "history.h"

namespace amunmt {

class Histories {
  public:
    //Histories() {} // for all histories in translation task
    Histories(const Sentences& sentences, bool normalizeScore);

    std::shared_ptr<History> at(size_t id) const {
      return coll_.at(id);
    }

    size_t size() const {
      return coll_.size();
    }

    void Add(const Beams& beams);
    void SortByLineNum();
    void Append(const Histories &other);
    Beam GetFirstHyps();
    void Output(const God &god) const;

  protected:
    std::vector<std::shared_ptr<History>> coll_;
    Histories(const Histories &) = delete;
};


}

