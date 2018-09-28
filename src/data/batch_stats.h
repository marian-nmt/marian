#pragma once

#include <deque>
#include <queue>

#include <boost/timer/timer.hpp>
#include "data/corpus.h"
#include "data/vocab.h"

namespace marian {
namespace data {

class BatchStats {
private:
  std::map<std::vector<size_t>, size_t> map_; // [(src len, tgt len)] -> batch size

public:
  BatchStats() { }

  size_t getBatchSize(const std::vector<size_t>& lengths) {
    auto it = map_.lower_bound(lengths);
    for(size_t i = 0; i < lengths.size(); ++i)
      while(it != map_.end() && it->first[i] < lengths[i])
        it++;

    ABORT_IF(it == map_.end(), "Missing batch statistics");
    return it->second;
  }

  void add(Ptr<data::CorpusBatch> batch, size_t multiplier = 1) {
    std::vector<size_t> lengths;
    for(size_t i = 0; i < batch->sets(); ++i)
      lengths.push_back((*batch)[i]->batchWidth());
    size_t batchSize = batch->size() * multiplier;

    if(map_[lengths] < batchSize)
      map_[lengths] = batchSize;
  }

  // helpers for multi-node
  // serialize into a flat vector, for MPI data exchange
  std::vector<size_t> flatten() const {
    std::vector<size_t> res;
    if(map_.empty())
      return res;
    auto numStreams = map_.begin()->first.size();
    // format:
    //  - num streams
    //  - tuples ((stream sizes), )
    res.push_back(numStreams);
    for (const auto& entry : map_) {
      ABORT_IF(entry.first.size() != numStreams, "inconsistent number of streams??");
      for (auto streamLen : entry.first)
        res.push_back(streamLen);
      res.push_back(entry.second);
    }
    return res;
  }

  // deserialize a flattened batchStats
  // used as part of MPI data exchange
  BatchStats(const std::vector<size_t>& flattenedStats) {
    if (flattenedStats.empty())
      return;
    size_t i = 0;
    auto numStreams = flattenedStats[i++];
    std::vector<size_t> lengths(numStreams);
    while (i < flattenedStats.size()) {
      for(auto& length : lengths)
        length = flattenedStats[i++];
      auto batchSize = flattenedStats[i++];
      map_[lengths] = batchSize;
    }
    ABORT_IF(i != flattenedStats.size(), "invalid flattenedVector??");
    //dump();
  }

  void dump() { // (for debugging)
    for (const auto& entry : map_) {
      for (auto streamLen : entry.first)
        std::cerr << streamLen << " ";
      std::cerr << ": " << entry.second << std::endl;
    }
  }
};
}  // namespace data
}  // namespace marian
