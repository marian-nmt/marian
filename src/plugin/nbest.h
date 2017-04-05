#pragma once

#include <string>
#include <vector>
#include <memory>


using NBestBatch = std::vector<std::vector<int>>;

namespace amunmt {
class Vocab;

class NBest {
  public:
    NBest(
      const std::vector<std::string>& nBestList,
      Vocab& trgVocab,
      const size_t maxBatchSize=64);

    std::vector<NBestBatch> SplitNBestListIntoBatches() const;

  protected:
    NBestBatch MaskAndTransposeBatch(const NBestBatch& batch) const;

  protected:
    std::vector<std::vector<int>> data_;
    const size_t maxBatchSize_;
};

}  // namespace amunmt;
