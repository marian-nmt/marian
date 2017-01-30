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
    std::shared_ptr<Histories> Process(const God &god, const Sentences& sentences);

    void PreProcess(
    		const God &god,
    		const Sentences& sentences,
    		std::shared_ptr<Histories> &histories,
    		Beam &prevHyps);

    void PostProcess();

    void Encode(const Sentences& sentences, States& states, States& nextStates);

    void Decode(
    		const God &god,
    		const Sentences& sentences,
    		States &states,
    		States &nextStates,
    		std::shared_ptr<Histories> &histories,
    		Beam &prevHyps);

    const DeviceInfo &GetDeviceInfo()
    { return deviceInfo_; }

  private:
    Search(const Search &) = delete;

    size_t MakeFilter(const God &god, const std::set<Word>& srcWords, size_t vocabSize);

    std::vector<ScorerPtr> scorers_;
    Words filterIndices_;
    BestHypsBasePtr bestHyps_;

    DeviceInfo deviceInfo_;
};

}

