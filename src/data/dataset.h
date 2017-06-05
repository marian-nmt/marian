#pragma once

#include "common/definitions.h"
#include "data/batch.h"
#include "data/vocab.h"

namespace marian {
namespace data {

template <class Sample, class Iterator, class Batch>
class DatasetBase {
protected:
  std::vector<std::string> paths_;

public:
  typedef Batch batch_type;
  typedef Ptr<Batch> batch_ptr;
  typedef Iterator iterator;
  typedef Sample sample;

  DatasetBase() {}
  DatasetBase(std::vector<std::string> paths) : paths_(paths) {}

  virtual Iterator begin() = 0;
  virtual Iterator end() = 0;
  virtual void shuffle() = 0;

  virtual batch_ptr toBatch(const std::vector<sample>&) = 0;

  virtual void reset() {}
  virtual void prepare() {}
};

typedef std::vector<float> Data;
typedef std::vector<Data> Example;
typedef std::vector<Example> Examples;

typedef Examples::const_iterator ExampleIterator;

class Input {
private:
  Shape shape_;
  Ptr<Data> data_;

public:
  typedef Data::iterator iterator;
  typedef Data::const_iterator const_iterator;

  /** @brief Constructs a new Input object with the specified Shape */
  Input(const Shape& shape)
      : shape_(shape), data_(new Data(shape_.elements(), 0.0f)) {}

  Data::iterator begin() { return data_->begin(); }
  Data::iterator end() { return data_->end(); }

  Data::const_iterator begin() const { return data_->cbegin(); }
  Data::const_iterator end() const { return data_->cend(); }

  /** @brief Returns a reference to this object's underlying ::Data. */
  Data& data() { return *data_; }

  /** @brief Gets this object's underlying Shape. */
  Shape shape() const { return shape_; }

  /** @brief Returns the number underlying values in this object's ::Data. */
  size_t size() const { return data_->size(); }
};

class DataBatch : public Batch {
private:
  std::vector<Input> inputs_;

public:
  std::vector<Input>& inputs() { return inputs_; }

  const std::vector<Input>& inputs() const { return inputs_; }

  void push_back(Input input) { inputs_.push_back(input); }

  Data& features() { return inputs_[0].data(); }

  Data& labels() { return inputs_.back().data(); }

  size_t size() const { return inputs_.front().shape()[0]; }
};

class Dataset : public DatasetBase<Example, ExampleIterator, DataBatch> {
protected:
  Examples examples_;

public:
  Dataset(std::vector<std::string> paths) : DatasetBase(paths) {}

  virtual void loadData() = 0;

  iterator begin() { return ExampleIterator(examples_.begin()); }

  iterator end() { return ExampleIterator(examples_.end()); }

  void shuffle() { std::random_shuffle(examples_.begin(), examples_.end()); }

  batch_ptr toBatch(const Examples& batchVector) {
    int batchSize = batchVector.size();

    std::vector<int> maxDims;
    for(auto& ex : batchVector) {
      if(maxDims.size() < ex.size())
        maxDims.resize(ex.size(), 0);
      for(size_t i = 0; i < ex.size(); ++i) {
        if(ex[i].size() > (size_t)maxDims[i])
          maxDims[i] = ex[i].size();
      }
    }

    batch_ptr batch(new DataBatch());
    std::vector<Input::iterator> iterators;
    for(auto& m : maxDims) {
      batch->push_back(Shape({batchSize, m}));
      iterators.push_back(batch->inputs().back().begin());
    }

    for(auto& ex : batchVector) {
      for(size_t i = 0; i < ex.size(); ++i) {
        Data d = ex[i];
        d.resize(maxDims[i], 0.0f);
        iterators[i] = std::copy(d.begin(), d.end(), iterators[i]);
      }
    }
    return batch;
  }
};
}
}
