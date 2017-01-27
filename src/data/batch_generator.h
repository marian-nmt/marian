#pragma once

#include <deque>
#include <queue>

#include <boost/timer/timer.hpp>

#include "dataset.h"

namespace marian {

namespace data {

template <class DataSet>
class BatchGenerator {
  public:
    typedef typename DataSet::batch_ptr BatchPtr;

    typedef typename DataSet::sample sample;
    typedef std::vector<sample> samples;

  private:
    Ptr<DataSet> data_;
    typename DataSet::iterator current_;

    size_t batchSize_;
    size_t maxiBatchSize_;

    std::deque<BatchPtr> bufferedBatches_;
    BatchPtr currentBatch_;

    void fillBatches() {
      auto cmp = [](const sample& a, const sample& b) {
        return a[0].size() < b[0].size();
      };

      std::priority_queue<sample, samples, decltype(cmp)> maxiBatch(cmp);

      while(current_ != data_->end() && maxiBatch.size() < maxiBatchSize_) {
        maxiBatch.push(*current_);
        current_++;
      }

      samples batchVector;
      while(!maxiBatch.empty()) {
        batchVector.push_back(maxiBatch.top());
        maxiBatch.pop();
        if(batchVector.size() == batchSize_) {
          bufferedBatches_.push_back(data_->toBatch(batchVector));
          batchVector.clear();
        }
      }
      if(!batchVector.empty())
        bufferedBatches_.push_back(data_->toBatch(batchVector));

      std::random_shuffle(bufferedBatches_.begin(), bufferedBatches_.end());
    }

  public:
    BatchGenerator(Ptr<DataSet> data,
                   size_t batchSize=80,
                   size_t maxiBatchNum=20)
    : data_(data),
      batchSize_(batchSize),
      maxiBatchSize_(batchSize * maxiBatchNum)
      { }

    operator bool() const {
      return !bufferedBatches_.empty();
    }

    BatchPtr next() {
      UTIL_THROW_IF2(bufferedBatches_.empty(),
                     "No batches to fetch, run prepare()");
      currentBatch_ = bufferedBatches_.front();
      bufferedBatches_.pop_front();

      if(bufferedBatches_.empty())
        fillBatches();

      return currentBatch_;
    }

    void prepare(bool shuffle=true) {
      if(shuffle)
        data_->shuffle();
      current_ = data_->begin();
      fillBatches();
    }
};

}

}
