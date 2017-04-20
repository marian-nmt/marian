#pragma once

#include <string>
#include <vector>
#include <memory>

#include "common/scorer.h"



using NBestBatch = std::vector<std::vector<int>>;

namespace amunmt {
class Vocab;

class RescoreBatch {
  public:
    RescoreBatch(std::vector<std::vector<int>>& data, std::vector<States*>& states);

    std::vector<std::vector<int>> data;
    std::vector<States*> states;
    std::vector<std::vector<std::pair<size_t, size_t>>> indices;
    std::vector<std::vector<size_t>> prevIds;
    std::vector<std::vector<size_t>> completed;

    size_t length() const;
    size_t size() const;

  protected:
    void ComputePrevIds();
    void ComputeIndices();

};

class NBest {
  public:
    NBest(
      const std::vector<std::string>& nBestList,
      std::vector<States>& states,
      Vocab& trgVocab,
      const size_t maxBatchSize=64);

    NBest(
      const std::vector<std::vector<std::string>>& nBestList,
      std::vector<States>& states,
      Vocab& trgVocab,
      const size_t maxBatchSize=64);

    std::vector<RescoreBatch> SplitNBestListIntoBatches() const;

  protected:
    void MaskAndTransposeBatch(NBestBatch& batch) const;

  protected:
    std::vector<std::vector<int>> data_;
    std::vector<States>& states_;
    const size_t maxBatchSize_;
};

}  // namespace amunmt;
