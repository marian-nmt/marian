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
  std::map<std::vector<size_t>, size_t> map_;

public:
  size_t getBatchSize(const std::vector<size_t>& lengths) {
    auto it = map_.lower_bound(lengths);
    for(int i = 0; i < lengths.size(); ++i)
      while(it != map_.end() && it->first[i] < lengths[i])
        it++;

    UTIL_THROW_IF2(it == map_.end(), "Missing batch statistics");
    return it->second;
  }

  void add(Ptr<data::CorpusBatch> batch) {
    std::vector<size_t> lengths;
    for(int i = 0; i < batch->sets(); ++i)
      lengths.push_back((*batch)[i]->batchWidth());
    size_t batchSize = batch->size();

    if(map_[lengths] < batchSize)
      map_[lengths] = batchSize;
  }
};
}
}
