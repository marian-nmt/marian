#pragma once

#include <memory>
#include "common/scorer.h"
#include "common/sentence.h"
#include "common/base_best_hyps.h"
#include "common/history.h"

namespace amunmt {

class Search {
  public:
    Search(const God &god);
    virtual ~Search();

    std::shared_ptr<Histories> Process(const God& god, const Sentences& sentences);

    const DeviceInfo& GetDeviceInfo() const;

    const std::vector<ScorerPtr>& GetScorers() const;

  protected:
    States NewStates() const;

    void PreProcess(
    		const Sentences& sentences,
    		std::shared_ptr<Histories>& histories,
    		Beam &prevHyps);

    void PostProcess();

    void Encode(const Sentences& sentences, States& states);

    void Decode(
    		const God &god,
    		const Sentences& sentences,
    		States& states,
    		std::shared_ptr<Histories>& histories,
    		Beam& prevHyps);

    size_t MakeFilter(const std::set<Word>& srcWords, size_t vocabSize);

    bool CalcBeam(
    		const God& god,
    		Beam& prevHyps,
    		Beams& beams,
    		std::vector<size_t>& beamSizes,
    		std::shared_ptr<Histories>& histories,
    		const Sentences& sentences,
    		States& states,
    		States& nextStates);

    Search(const Search&) = delete;

  protected:
    std::vector<ScorerPtr> scorers_;
    std::shared_ptr<const Filter> filter_;
    const size_t maxBeamSize_;
    Words filterIndices_;
    BestHypsBasePtr bestHyps_;

    bool returnAlignment_;

    DeviceInfo deviceInfo_;
};

}

