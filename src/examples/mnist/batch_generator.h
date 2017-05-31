#pragma once

#include <deque>
#include <queue>

#include <boost/timer/timer.hpp>

#include "examples/mnist/dataset.h"


namespace marian {
namespace data {

class Input {
  private:
    Shape shape_;
    DataPtr data_;

  public:
    typedef Data::iterator iterator;
    typedef Data::const_iterator const_iterator;

    /** @brief Constructs a new Input object with the specified Shape */
    Input(const Shape& shape)
    : shape_(shape),
      data_(new Data(shape_.elements(), 0.0f)) {}

    Data::iterator begin() {
      return data_->begin();
    }
    Data::iterator end() {
      return data_->end();
    }

    Data::const_iterator begin() const {
      return data_->cbegin();
    }
    Data::const_iterator end() const {
      return data_->cend();
    }

    /** @brief Returns a reference to this object's underlying ::Data. */
    Data& data() {
      return *data_;
    }

    /** @brief Gets this object's underlying Shape. */
    Shape shape() const {
      return shape_;
    }

    /** @brief Returns the number underlying values in this object's ::Data. */
    size_t size() const {
      return data_->size();
    }
};

class Batch {
  private:
    std::vector<Input> inputs_;

  public:
    std::vector<Input>& inputs() {
      return inputs_;
    }

    const std::vector<Input>& inputs() const {
      return inputs_;
    }

    void push_back(Input input) {
      inputs_.push_back(input);
    }

    int dim() const {
      return inputs_[0].shape()[0];
    }

    size_t size() const {
      //return inputs_.size();
      return dim();
    }
};

typedef std::shared_ptr<Batch> BatchPtr;


template <class DataSet>
class MNISTBatchGenerator {
  public:
    typedef typename DataSet::batch_ptr BatchPtr;

  private:
    Ptr<DataSet> data_;
    ExampleIterator current_;

    size_t batchSize_;
    size_t maxiBatchSize_;

    std::deque<BatchPtr> bufferedBatches_;
    BatchPtr currentBatch_;

    void fillBatches() {
      auto cmp = [](const ExamplePtr& a, const ExamplePtr& b) {
        return (*a)[0]->size() < (*b)[0]->size();
      };

      std::priority_queue<ExamplePtr, Examples, decltype(cmp)> maxiBatch(cmp);

      while(current_ != data_->end() && maxiBatch.size() < maxiBatchSize_) {
        maxiBatch.push(*current_);
        current_++;
      }

      Examples batchVector;
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
    MNISTBatchGenerator(Ptr<DataSet> data, size_t batchSize=80, size_t maxiBatchNum=20)
    : data_(data),
      batchSize_(batchSize),
      maxiBatchSize_(batchSize * maxiBatchNum),
      current_(data_->begin()) { }

    operator bool() const {
      return !bufferedBatches_.empty();
    }

    BatchPtr next() {
      UTIL_THROW_IF2(bufferedBatches_.empty(), "No batches to fetch");
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
