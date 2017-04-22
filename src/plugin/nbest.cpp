#include "nbest.h"

#include <algorithm>
#include <iomanip>

#include "common/utils.h"
#include "common/vocab.h"
#include "common/types.h"

namespace amunmt {

size_t RescoreBatch::length() const {
  return data.size();
}

size_t RescoreBatch::size() const {
  return data[0].size();
}

RescoreBatch::RescoreBatch(std::vector<std::vector<int>>& data, std::vector<States*>& states)
  : data(data),
    states(states)
{
    ComputePrevIds();
    ComputeIndices();
}


void RescoreBatch::ComputePrevIds() {
  std::vector<size_t> ids;
  for (size_t i = 0; i < size(); ++i) {
    ids.push_back(i);
  }

  prevIds.push_back(ids);

  for (size_t step = 0; step < length() - 1; ++step) {
    std::vector<size_t> nextIds;
    std::vector<size_t> compIds;
    for (size_t i = 0; i < prevIds[step].size(); ++i) {
      if (data[step + 1][prevIds[step][i]] != -1) {
        nextIds.push_back(prevIds[step][i]);
      } else {
        compIds.push_back(i);
      }
    }
    prevIds.emplace_back(std::move(nextIds));
    completed.emplace_back(std::move(compIds));
  }
}

void RescoreBatch::ComputeIndices() {
  for (size_t step = 0; step < data.size(); ++step) {
    std::vector<std::pair<size_t, size_t>> ids;
    for (size_t i = 0; i < prevIds[step].size(); ++i) {
      ids.push_back(std::make_pair(i, data[step][prevIds[step][i]]));
    }
    indices.emplace_back(std::move(ids));
  }
}


NBest::NBest(
  const std::vector<std::string>& nBestList,
  std::vector<States>& states,
  Vocab& trgVocab,
  const size_t maxBatchSize)
    : states_(states),
      maxBatchSize_(maxBatchSize)
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

NBest::NBest(
  const std::vector<std::vector<std::string>>& nBestList,
  std::vector<States>& states,
  Vocab& trgVocab,
  const size_t maxBatchSize)
    : maxBatchSize_(maxBatchSize),
      states_(states)
{
  for (auto& tokens : nBestList) {
    std::vector<int> ids;
    for (auto token : trgVocab(tokens, false)) {
      ids.push_back((int)token);
    }
    data_.push_back(ids);
  }
}

std::vector<RescoreBatch> NBest::SplitNBestListIntoBatches() const {
  std::vector<RescoreBatch> batches;
  std::vector<std::vector<int>> sBatch;
  std::vector<States*> inputStates;
  for (size_t i = 0; i < data_.size(); ++i) {
    sBatch.push_back(data_[i]);
    if (states_.size() == 1) {
      inputStates.push_back(&states_[0]);
    } else {
      inputStates.push_back(&states_[i]);
    }

    if (sBatch.size() == maxBatchSize_ || i == data_.size() - 1) {
      MaskAndTransposeBatch(sBatch);
      batches.emplace_back(sBatch, inputStates);
      sBatch.clear();
      inputStates.clear();
    }
  }
  return batches;
}


inline void NBest::MaskAndTransposeBatch(NBestBatch& batch) const {
  size_t maxLength = 0;
  for (auto& sentence: batch) {
    maxLength = std::max(maxLength, sentence.size());
  }
  maxLength++;

  NBestBatch masked;
  for (size_t i = 0; i < maxLength; ++i) {
    masked.emplace_back(batch.size(), -1);
    for (size_t j = 0; j < batch.size(); ++j) {
      if (i < batch[j].size()) {
        masked[i][j] = batch[j][i];
      }
    }
  }
  std::swap(batch, masked);
}

}  // namespace amunmt
