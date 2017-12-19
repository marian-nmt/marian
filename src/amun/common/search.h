#pragma once

#include <memory>
#include <set>

#include "common/scorer.h"
#include "common/sentence.h"
#include "common/base_best_hyps.h"

namespace amunmt {

class Histories;
class FilterVocab;
class Sentences;
using SentencesPtr = std::shared_ptr<Sentences>;

class Search {
  public:
    Search(const God &god);
    virtual ~Search();

    void Translate(SentencesPtr sentences);

    BestHypsBase &GetBestHyps() const
    { return *bestHyps_; }

    std::shared_ptr<const FilterVocab> GetFilter() const
    { return filter_; }

    const Words &GetFilterIndices() const
    { return filterIndices_; }

    bool NormalizeScore() const
    { return normalizeScore_; }

    void FilterTargetVocab(const Sentences& sentences);

  protected:

    Search(const Search&) = delete;

    DeviceInfo deviceInfo_;
    std::vector<ScorerPtr> scorers_;
    std::shared_ptr<const FilterVocab> filter_;
    bool normalizeScore_;
    Words filterIndices_;
    BestHypsBasePtr bestHyps_;
};

}

