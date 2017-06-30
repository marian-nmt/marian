#pragma once

#include <memory>
#include <set>

#include "common/scorer.h"
#include "common/sentence.h"
#include "common/base_best_hyps.h"

namespace amunmt {

class Histories;
class Filter;

class Search {
  public:
    Search(const God &god);
    virtual ~Search();

    std::shared_ptr<Histories> Translate(const Sentences& sentences);

  protected:
    States NewStates() const;
    void FilterTargetVocab(const Sentences& sentences);
    States Encode(const Sentences& sentences);
    void CleanAfterTranslation();

    bool CalcBeam(
    		std::shared_ptr<Histories>& histories,
    		std::vector<uint>& beamSizes,
        Beam& prevHyps,
    		States& states,
    		States& nextStates);

    Search(const Search&) = delete;

  protected:
    DeviceInfo deviceInfo_;
    std::vector<ScorerPtr> scorers_;
    std::shared_ptr<const Filter> filter_;
    const size_t maxBeamSize_;
    bool normalizeScore_;
    Words filterIndices_;
    BestHypsBasePtr bestHyps_;
};

}

