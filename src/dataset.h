#pragma once

// This file is part of the Marian toolkit.
// Marian is copyright (c) 2016 Marcin Junczys-Dowmunt.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <memory>

namespace marian {
namespace data {

typedef std::vector<float> Data;
typedef std::shared_ptr<Data> DataPtr;

typedef std::vector<DataPtr> Example;
typedef std::shared_ptr<Example> ExamplePtr;

typedef std::vector<ExamplePtr> Examples;

class Input {
  private:
    Shape shape_;
    DataPtr data_;

  public:
    typedef Data::iterator iterator;
    typedef Data::const_iterator const_iterator;

    Input(const Shape& shape)
    : shape_(shape),
      data_(new Data(shape_.totalSize(), 0.0f)) {}

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

    Shape shape() const {
      return shape_;
    }

    size_t size() const {
      return data_->size();
    }
};

typedef Examples::const_iterator ExampleIterator;

class DataBase {
  public:
    virtual ExampleIterator begin() const = 0;
    virtual ExampleIterator end() const = 0;
    virtual void shuffle() = 0;

    virtual int dim(size_t i) {
      return (*begin())->at(i)->size();
    }
};

typedef std::shared_ptr<DataBase> DataBasePtr;

template <class Set, typename ...Args>
DataBasePtr DataSet(Args ...args) {
  return DataBasePtr(new Set(args...));
}

}
}
