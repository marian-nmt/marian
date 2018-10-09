#pragma once

#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <queue>

#include "common/config.h"
#include "data/batch_stats.h"
#include "data/rng_engine.h"
#include "training/training_state.h"

namespace marian {
namespace data {

template <class DataSet>
class BatchGenerator : public RNGEngine {
public:
  typedef typename DataSet::batch_ptr BatchPtr;

  typedef typename DataSet::sample sample;
  typedef std::vector<sample> samples;     // @TODO: type names should be capitalized

protected:
  Ptr<DataSet> data_;
  Ptr<Config> options_;
  bool restored_{false};

private:
  Ptr<BatchStats> stats_;

  int batchSize_{1};

  typename DataSet::iterator current_;
  bool newlyPrepared_{true};

  size_t maxiBatchSize_;
  std::deque<BatchPtr> bufferedBatches_;
  BatchPtr currentBatch_;

  mutable std::mutex loadMutex_;
  mutable std::condition_variable loadCondition_;
  bool loadReady_{true};

  // this runs on a bg thread; sequencing is handled by caller, but locking is done in here
  void fillBatches(bool shuffle = true) {
    LOG(info, "fillBatches entered");
    typedef typename sample::value_type Item;
    auto itemCmp = [](const Item& sa, const Item& sb) { return sa.size() < sb.size(); }; // sort by element length, not content

    auto cmpSrc = [itemCmp](const sample& a, const sample& b) {
      return std::lexicographical_compare(
          a.begin(), a.end(), b.begin(), b.end(), itemCmp);
    };

    auto cmpTrg = [itemCmp](const sample& a, const sample& b) {
      return std::lexicographical_compare(
          a.rbegin(), a.rend(), b.rbegin(), b.rend(), itemCmp);
    };

    auto cmpNone = [](const sample& a, const sample& b) { return &a < &b; }; // instead sort by address, so we have something to work with

    typedef std::function<bool(const sample&, const sample&)> cmp_type;
    typedef std::priority_queue<sample, samples, cmp_type> sample_queue;

    std::unique_ptr<sample_queue> maxiBatch; // priority queue, shortest first

    if(options_->has("maxi-batch-sort")) {
      if(options_->get<std::string>("maxi-batch-sort") == "src")
        maxiBatch.reset(new sample_queue(cmpSrc));
      else if(options_->get<std::string>("maxi-batch-sort") == "none")
        maxiBatch.reset(new sample_queue(cmpNone));
      else
        maxiBatch.reset(new sample_queue(cmpTrg));
    } else {
      maxiBatch.reset(new sample_queue(cmpNone));
    }

    size_t maxBatchSize = options_->get<int>("mini-batch");
    size_t maxSize = maxBatchSize * options_->get<int>("maxi-batch");

    // consume data from corpus into maxi-batch (single sentences)
    // sorted into specified order (due to queue)
    if(newlyPrepared_) {
      current_ = data_->begin();
      newlyPrepared_ = false;
    } else {
      if(current_ != data_->end())
        ++current_;
    }
    size_t sets = 0;
    try {
    LOG(info, "begin read lines, current size {}", maxiBatch->size());
    while(current_ != data_->end() && maxiBatch->size() < maxSize) { // loop over data
      maxiBatch->push(*current_);
      sets = current_->size();
      // do not consume more than required for the maxi batch as this causes
      // that line-by-line translation is delayed by one sentence
      bool last = maxiBatch->size() == maxSize;
      if(!last)
        ++current_; // this actually reads the next line and pre-processes it
    }
    LOG(info, "end read lines, current size {}", maxiBatch->size());
    // @TODO: Consider using MPI at this point to parallelize parsing.
    }
    catch (const std::exception & e) {
      LOG("exception caught while reading: {}", e.what());
      logCallStack(0);
      throw;
    }

    // construct the actual batches and place them in the queue
    samples batchVector;
    size_t currentWords = 0;
    std::vector<size_t> lengths(sets, 0); // records maximum length observed within current batch

    std::vector<BatchPtr> tempBatches;
    tempBatches.reserve(10000); // (should be enough in most cases; not critical)

    // process all loaded sentences in order of increasing length
    // @TODO: we could just use a vector and do a sort() here; would make the cost more explicit
    LOG(info, "begin form batches, #batches = {}", maxiBatch->size());
    const size_t mbWords = options_->get<size_t>("mini-batch-words", 0);
    const bool useDynamicBatching = options_->has("mini-batch-fit");
    BatchStats::const_iterator cachedStatsIter;
    if (stats_)
      cachedStatsIter = stats_->begin();
    while(!maxiBatch->empty()) { // while there are sentences in the queue
      // push item onto batch
      batchVector.push_back(maxiBatch->top());
      maxiBatch->pop(); // fetch next-shortest

      // have we reached sufficient amount of data to form a batch?
      bool makeBatch;
      if(useDynamicBatching) { // batch size based on dynamic batching
        if(stats_) {
          for(size_t i = 0; i < sets; ++i)
            if(batchVector.back()[i].size() > lengths[i])
              lengths[i] = batchVector.back()[i].size(); // record max lengths so far

          maxBatchSize = stats_->findBatchSize(lengths, cachedStatsIter);
#if 1     // sanity check
          auto it = stats_->lower_bound(lengths);
          auto maxBatchSize1 = stats_->findBatchSize(lengths, it);
          ABORT_IF(maxBatchSize != maxBatchSize1, "findBatchSize iter caching logic is borked");
#endif

          makeBatch = batchVector.size() >= maxBatchSize;
          // if last added sentence caused a bump then we likely have bad padding, so rather move it into the next batch
          if(batchVector.size() > maxBatchSize) {
            maxiBatch->push(batchVector.back());
            batchVector.pop_back();
          }
        }
      }
      else if(mbWords > 0) {
        currentWords += batchVector.back()[0].size(); // count words based on first stream =source  --@TODO: shouldn't we count based on labels?
        makeBatch = currentWords > mbWords; // Batch size based on sentences
      }
      else
        makeBatch = batchVector.size() == maxBatchSize; // Batch size based on words

      // if we reached the desired batch size then create a real batch
      if(makeBatch) {
        tempBatches.push_back(data_->toBatch(batchVector));

        // prepare for next batch
        batchVector.clear();
        currentWords = 0;
        lengths.assign(sets, 0);
        if (stats_)
          cachedStatsIter = stats_->begin();
      }
    }

    // turn rest into batch
    if(!batchVector.empty())
      tempBatches.push_back(data_->toBatch(batchVector));
    LOG(info, "end form batches, #tempBatches = {}", tempBatches.size());

    if(shuffle) {
      // shuffle the batches
      std::shuffle(tempBatches.begin(), tempBatches.end(), eng_);
    }
    LOG(info, "end shuffling batches, #tempBatches = {}", tempBatches.size());

    // put batches onto queue
    // exclusive lock
    std::unique_lock<std::mutex> lock(loadMutex_);
    LOG(info, "begin pushing batches (this is after lock), #tempBatches = {}", tempBatches.size());
    for(const auto& batch : tempBatches) // @TODO: use insert()
      bufferedBatches_.push_back(batch);
    LOG(info, "fillBatches completed, bufferedBatches.size = {}", bufferedBatches_.size());
  }

public:
  BatchGenerator(Ptr<DataSet> data,
                 Ptr<Config> options,
                 Ptr<BatchStats> stats = nullptr)
      : data_(data), options_(options), stats_(stats) {}

  operator bool() const {
#if 0
    // wait if empty but loading
    std::unique_lock<std::mutex> lock(loadMutex_);
    loadCondition_.wait(
        lock, [this] { return loadReady_ || !bufferedBatches_.empty(); });
#endif

    return !bufferedBatches_.empty();
  }

  BatchPtr next() {
#if 1 // not threaded  --note: also disable in operator bool()
    ABORT_IF(bufferedBatches_.empty(), "No batches to fetch, run prepare()");
    currentBatch_ = bufferedBatches_.front();
    bufferedBatches_.pop_front();
    if (bufferedBatches_.empty())
      fillBatches();
#else
    {
      std::unique_lock<std::mutex> lock(loadMutex_);
      loadCondition_.wait(
          lock, [this] { return loadReady_ || !bufferedBatches_.empty(); });
      // @TODO: same code as operator bool()
    }

    ABORT_IF(bufferedBatches_.empty(), "No batches to fetch, run prepare()");
    currentBatch_ = bufferedBatches_.front();

    if(loadReady_
      && (int)bufferedBatches_.size()
#if 1
              <= 5  // @TODO: only preroll one, to see if we have a threading issue
#else
              <= 100/*std::max(options_->get<int>("maxi-batch") / 5, 1)*/  // @TODO: rather, pull Marcin's proper fix
#endif
        ) {
      {
        std::unique_lock<std::mutex> lock(loadMutex_);
        loadReady_ = false;
        loadCondition_.notify_all();
      }

      std::thread([this]() {
        fillBatches();
        std::unique_lock<std::mutex> lock(loadMutex_);
        loadReady_ = true;
        loadCondition_.notify_all();
      })
      .detach();
    }

    std::unique_lock<std::mutex> lock(loadMutex_);
    bufferedBatches_.pop_front();
#endif

    return currentBatch_;
  }

  std::vector<BatchPtr> nextN(size_t num) {
    std::vector<BatchPtr> batches;
    for(int i = 0; i < num && *this; ++i)
      batches.push_back(next());
    return batches;
  }

  void prepare(bool shuffle = true) {
    if(shuffle)
      data_->shuffle();
    else
      data_->reset();
    newlyPrepared_ = true;
    fillBatches(shuffle);
  }

  bool restore(Ptr<TrainingState> state, bool shuffle) {
    if(state->epochs == 1 && state->batchesEpoch == 0)
      return false;

    LOG(info,
        "[data] Restoring the corpus state to epoch {}, batch {}",
        state->epochs,
        state->batches);

    if(state->epochs > 1) {
      data_->restore(state);
      setRNGState(state->seedBatch);
    }

    prepare(shuffle);
    for(size_t i = 0; i < state->batchesEpoch; ++i)
      next();

    return true;
  }
};

class CorpusBatchGenerator : public BatchGenerator<CorpusBase>,
                             public TrainingObserver {
public:
  CorpusBatchGenerator(Ptr<CorpusBase> data,
                       Ptr<Config> options,
                       Ptr<BatchStats> stats = nullptr)
      : BatchGenerator(data, options, stats) {}

  void actAfterEpoch(TrainingState& state) override {
    state.seedBatch = getRNGState();
    state.seedCorpus = data_->getRNGState();
  }
};
}  // namespace data
}  // namespace marian
