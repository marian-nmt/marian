#pragma once

#include <memory>
#include <set>

#include "common/scorer.h"
#include "common/sentence.h"
#include "common/base_best_hyps.h"

namespace amunmt {

class Histories;
class FilterVocab;

class Search {
  public:
    Search(const God &god);
    virtual ~Search();

    std::shared_ptr<Histories> Translate(const Search &search, const Sentences& sentences);

    std::shared_ptr<const FilterVocab> GetFilter() const
    { return filter_; }

    const Words &GetFilterIndices() const
    { return filterIndices_; }

  protected:
    States NewStates() const;
    void FilterTargetVocab(const Sentences& sentences);
    States Encode(const Sentences& sentences);
    void CleanAfterTranslation();

    Search(const Search&) = delete;

    DeviceInfo deviceInfo_;
    std::vector<ScorerPtr> scorers_;
    std::shared_ptr<const FilterVocab> filter_;
    const size_t maxBeamSize_;
    bool normalizeScore_;
    Words filterIndices_;
    BestHypsBasePtr bestHyps_;
};

}

