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

class MNIST : public Dataset {
private:
  const int IMAGE_MAGIC_NUMBER;
  const int LABEL_MAGIC_NUMBER;

public:
  MNIST(std::vector<std::string> paths,
        std::vector<Ptr<Vocab>> vocabs = {},
        Ptr<Config> options = nullptr)
      : Dataset(paths), IMAGE_MAGIC_NUMBER(2051), LABEL_MAGIC_NUMBER(2049) {
    loadData();
  }

  void loadData() {
    UTIL_THROW_IF2(paths_.size() != 2,
                   "Paths to MNIST data files are not specified");

    auto features = ReadImages(paths_[0]);
    auto labels = ReadLabels(paths_[1]);
    UTIL_THROW_IF2(features.size() != labels.size(),
                   "Features do not match labels");

    for(size_t i = 0; i < features.size(); ++i) {
      Example ex = {features[i], labels[i]};
      examples_.emplace_back(ex);
    }
  }

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
    UTIL_THROW_IF2(!file.is_open(), "Cannot open file `" + full_path + "`!");

    int magic_number = 0;
    file.read((char *)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);

    UTIL_THROW_IF2(magic_number != IMAGE_MAGIC_NUMBER,
                   "Invalid MNIST image file!");

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
}
}
