#pragma once

#include <deque>
#include <queue>

#include <boost/timer/timer.hpp>

#include "data/dataset.h"
#include "data/batch_stats.h"
#include "data/vocab.h"
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
    Ptr<BatchStats> stats_;

    typename DataSet::iterator current_;

    size_t maxiBatchSize_;

    std::deque<BatchPtr> bufferedBatches_;
    BatchPtr currentBatch_;

    void fillBatches(bool shuffle=true) {
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
      int currentWords = 0;
      size_t sets = 2;
      std::vector<size_t> lengths(sets, 0);
      int maxBatchSize = options_->get<int>("mini-batch");
      
      while(!maxiBatch.empty()) {
        batchVector.push_back(maxiBatch.top());
        currentWords += batchVector.back()[0].size();
        maxiBatch.pop();
        
        // Batch size based on sentences
        bool makeBatch = batchVector.size() == maxBatchSize;
        
        // Batch size based on words
        if(options_->has("mini-batch-words")) {
          int mbWords = options_->get<int>("mini-batch-words");
          if(mbWords > 0)
            makeBatch = currentWords > mbWords;
        }
        
        if(options_->has("dynamic-batching")) {
          // Dynamic batching
          if(stats_ && options_->get<bool>("dynamic-batching")) {
            for(size_t i = 0; i < sets; ++i)
              if(batchVector.back()[i].size() > lengths[i])
                lengths[i] = batchVector.back()[i].size();
            
            maxBatchSize = stats_->getBatchSize(lengths);
            
            if(batchVector.size() > maxBatchSize) {
              maxiBatch.push(batchVector.back());
              batchVector.pop_back();
              makeBatch = true;
            }
            else {
              makeBatch = batchVector.size() == maxBatchSize;
            }
          }
        }
        
        if(makeBatch) {
          //std::cerr << "Creating batch" << std::endl;
          bufferedBatches_.push_back(data_->toBatch(batchVector));
          batchVector.clear();
          currentWords = 0;
          lengths.clear();
          lengths.resize(sets, 0);
        }
      }
      if(!batchVector.empty())
        bufferedBatches_.push_back(data_->toBatch(batchVector));

      if(shuffle) {
        std::random_shuffle(bufferedBatches_.begin(), bufferedBatches_.end());
      }
    }

  public:
    BatchGenerator(Ptr<DataSet> data,
                   Ptr<Config> options,
                   Ptr<BatchStats> stats = nullptr)
    : data_(data),
      options_(options),
      stats_(stats) { }

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
      else
        data_->reset();
      current_ = data_->begin();
      fillBatches(shuffle);
    }
};

}

}
