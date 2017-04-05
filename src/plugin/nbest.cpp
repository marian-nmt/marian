#include "nbest.h"

#include <algorithm>

#include "common/utils.h"
#include "common/vocab.h"
#include "common/types.h"

namespace amunmt {

NBest::NBest(
  const std::vector<std::string>& nBestList,
  Vocab& trgVocab,
  const size_t maxBatchSize)
    : maxBatchSize_(maxBatchSize)
{
  for (size_t i = 0; i < nBestList.size(); ++i) {
    std::vector<std::string> tokens;
    Split(nBestList[i], tokens);
    std::vector<int> ids;
    for (auto token : trgVocab(tokens)) {
      ids.push_back((int)token);
    }
    data_.push_back(ids);
  }
}


std::vector<NBestBatch> NBest::SplitNBestListIntoBatches() const {
  std::vector<NBestBatch> batches;
  std::vector<std::vector<int>> sBatch;
  for (size_t i = 0; i < data_.size(); ++i) {
    sBatch.push_back(data_[i]);
    if (sBatch.size() == maxBatchSize_ || i == data_.size() - 1) {
      batches.push_back(MaskAndTransposeBatch(sBatch));
      sBatch.clear();
    }
  }
  return batches;
}


inline NBestBatch NBest::MaskAndTransposeBatch(const NBestBatch& batch) const {
  size_t maxLength = 0;
  for (auto& sentence: batch) {
    maxLength = std::max(maxLength, sentence.size());
  }

  NBestBatch masked;
  for (size_t i = 0; i < maxLength; ++i) {
    masked.emplace_back(batch.size(), -1);
    for (size_t j = 0; j < batch.size(); ++j) {
      if (i < batch[j].size()) {
        masked[i][j] = batch[j][i];
      }
    }
  }
  return masked;
}

}  // namespace amunmt
