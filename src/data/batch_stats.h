#pragma once

#include <deque>
#include <queue>

#include "data/corpus.h"
#include "data/vocab.h"

namespace marian {
namespace data {

class BatchStats {
private:
  std::map<std::vector<size_t>, size_t> map_; // [(src len, tgt len)] -> batch size

public:
  BatchStats() { }

  typedef std::map<std::vector<size_t>, size_t>::const_iterator const_iterator;
  const_iterator begin() const { return map_.begin(); }
  const_iterator lower_bound(const std::vector<size_t>& lengths) const { return map_.lower_bound(lengths); }

  size_t findBatchSize(const std::vector<size_t>& lengths, const_iterator& it) const {
    // find the first item where all item.first[i] >= lengths[i], i.e. that can fit sentence tuples of lengths[]
    // This is expected to be called multiple times with increasing sentence lengths.
    // To get an initial value for 'it', call lower_bound() or begin().

    bool done = false;
    while (!done && it != map_.end()) {
      done = true;
      for(size_t i = 0; i < lengths.size(); ++i)
        while(it != map_.end() && it->first[i] < lengths[i]) {
          it++;
          done = false; // it++ might have decreased a key[<i], so we must check once again
        }
    }

    ABORT_IF(it == map_.end(), "Missing batch statistics");
    return it->second;
  }

  void add(Ptr<data::CorpusBatch> batch, double multiplier = 1.) {
    std::vector<size_t> lengths;
    for(size_t i = 0; i < batch->sets(); ++i)
      lengths.push_back((*batch)[i]->batchWidth());
    size_t batchSize = (size_t)ceil((double)batch->size() * multiplier);

    if(map_[lengths] < batchSize)
      map_[lengths] = batchSize;
  }

  // return a rough minibatch size in labels
  // We average over all (batch sizes * max trg length).
  size_t estimateTypicalTrgWords() const {
    size_t sum = 0;
    for (const auto& entry : map_) {
      auto maxTrgLength = entry.first.back();
      auto numSentences = entry.second;
      sum += numSentences * maxTrgLength;
    }
    return sum / map_.size();
  }

  // helpers for multi-node  --note: presently unused, but keeping them around for later use
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
