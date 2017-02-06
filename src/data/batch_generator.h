#pragma once

#include <deque>
#include <queue>

#include <boost/timer/timer.hpp>

#include "data/dataset.h"
#include "training/config.h"

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
    Ptr<Config> options_;

    typename DataSet::iterator current_;

    size_t maxiBatchSize_;

    std::deque<BatchPtr> bufferedBatches_;
    BatchPtr currentBatch_;

    void fillBatches() {
      auto cmp = [](const sample& a, const sample& b) {
        return a[0].size() < b[0].size();
      };

      std::priority_queue<sample, samples, decltype(cmp)> maxiBatch(cmp);

      int maxSize = options_->get<int>("mini-batch") * options_->get<int>("maxi-batch");
      while(current_ != data_->end() && maxiBatch.size() < maxSize) {
        maxiBatch.push(*current_);
        current_++;
      }

      samples batchVector;
      while(!maxiBatch.empty()) {
        batchVector.push_back(maxiBatch.top());
        maxiBatch.pop();
        if(batchVector.size() == options_->get<int>("mini-batch")) {
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
                   Ptr<Config> options)
    : data_(data),
      options_(options) { }

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
