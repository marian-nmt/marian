#pragma once

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "common/definitions.h"
#include "common/shape.h"
#include "data/batch.h"
#include "data/dataset.h"
#include "data/vocab.h"

namespace marian {
namespace data {

typedef std::vector<float> Data;
typedef std::vector<IndexType> Labels;
struct Example : public std::vector<Data> { // a std::vector<Data> with a getId() method
  size_t id;
  size_t getId() const { return id; }
  Example(std::vector<Data>&& data, size_t id) : std::vector<Data>(std::move(data)), id(id) {}
  Example() : id(SIZE_MAX) {}
};
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
  Input(const Shape& shape) : shape_(shape), data_(new Data(shape_.elements(), 0.0f)) {}

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

  virtual std::vector<Ptr<Batch>> split(size_t /*n*/, size_t /*sizeLimit*/) override { ABORT("Not implemented"); }

  Data& features() { return inputs_[0].data(); }

  Data& labels() { return inputs_.back().data(); }

  size_t size() const override { return inputs_.front().shape()[0]; }

  void setGuidedAlignment(std::vector<float>&&) override {
    ABORT("Guided alignment in DataBatch is not implemented");
  }
  void setDataWeights(const std::vector<float>&) override {
    ABORT("Data weighting in DataBatch is not implemented");
  }
};

class Dataset : public DatasetBase<Example, ExampleIterator, DataBatch>, public RNGEngine {
protected:
  Examples examples_;

public:
  Dataset(const std::vector<std::string>& paths, Ptr<Options> options)
      : DatasetBase(paths, options) {}

  virtual void loadData() = 0;

  iterator begin() override { return ExampleIterator(examples_.begin()); }

  iterator end() override { return ExampleIterator(examples_.end()); }

  void shuffle() override { std::shuffle(examples_.begin(), examples_.end(), eng_); }

  batch_ptr toBatch(const Examples& batchVector) override {
    int batchSize = (int)batchVector.size();

    std::vector<int> maxDims;
    for(auto& ex : batchVector) {
      if(maxDims.size() < ex.size())
        maxDims.resize(ex.size(), 0);
      for(size_t i = 0; i < ex.size(); ++i) {
        if(ex[i].size() > (size_t)maxDims[i])
          maxDims[i] = (int)ex[i].size();
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

class MNISTData : public Dataset {
private:
  const int IMAGE_MAGIC_NUMBER;
  const int LABEL_MAGIC_NUMBER;

public:
  MNISTData(std::vector<std::string> paths,
            std::vector<Ptr<Vocab>> /*vocabs*/ = {},
            Ptr<Options> options = nullptr)
      : Dataset(paths, options), IMAGE_MAGIC_NUMBER(2051), LABEL_MAGIC_NUMBER(2049) {
    loadData();
  }

  virtual ~MNISTData(){}

  void loadData() override {
    ABORT_IF(paths_.size() != 2, "Paths to MNIST data files are not specified");

    auto features = ReadImages(paths_[0]);
    auto labels = ReadLabels(paths_[1]);
    ABORT_IF(features.size() != labels.size(), "Features do not match labels");

    for(size_t i = 0; i < features.size(); ++i) {
      examples_.emplace_back(std::vector<Data>{ features[i], labels[i] }, i);
    }
  }

  Example next() override { return Example(); } //@TODO: this return was added to fix a warning. Is it correct?

private:
  typedef unsigned char uchar;

  int reverseInt(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255,
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
  }

  std::vector<Data> ReadImages(const std::string &full_path) {
    std::ifstream file(full_path);
    ABORT_IF(!file.is_open(), "Cannot open file `" + full_path + "`!");

    int magic_number = 0;
    file.read((char *)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);

    ABORT_IF(magic_number != IMAGE_MAGIC_NUMBER, "Invalid MNIST image file!");

    int number_of_images;
    int n_rows = 0;
    int n_cols = 0;

    file.read((char *)&number_of_images, sizeof(number_of_images));
    number_of_images = reverseInt(number_of_images);
    file.read((char *)&n_rows, sizeof(n_rows));
    n_rows = reverseInt(n_rows);
    file.read((char *)&n_cols, sizeof(n_cols));
    n_cols = reverseInt(n_cols);

    int imgSize = n_rows * n_cols;
    std::vector<Data> dataset(number_of_images);
    for(int i = 0; i < number_of_images; ++i) {
      dataset[i] = Data(imgSize, 0);
      for(int j = 0; j < imgSize; j++) {
        unsigned char pixel = 0;
        file.read((char *)&pixel, sizeof(pixel));
        dataset[i][j] = pixel / 255.0f;
      }
    }
    return dataset;
  }

  std::vector<Data> ReadLabels(const std::string &full_path) {
    std::ifstream file(full_path);

    if(!file.is_open())
      throw std::runtime_error("Cannot open file `" + full_path + "`!");

    int magic_number = 0;
    file.read((char *)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);

    if(magic_number != LABEL_MAGIC_NUMBER)
      throw std::runtime_error("Invalid MNIST label file!");

    int number_of_labels;
    file.read((char *)&number_of_labels, sizeof(number_of_labels));
    number_of_labels = reverseInt(number_of_labels);

    std::vector<Data> dataset(number_of_labels);
    for(int i = 0; i < number_of_labels; i++) {
      dataset[i] = Data(1, 0.0f);
      unsigned char label;
      file.read((char *)&label, 1);
      dataset[i][0] = label;
    }

    return dataset;
  }
};
}  // namespace data
}  // namespace marian
