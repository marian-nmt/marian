#pragma once

// This file is part of the Marian toolkit.

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


/**
 * @brief Namespace for code related to managing user data in Marian
 */
namespace data {

/** @brief Defines a convenience type to represent an ordered collection of floating point data. */
typedef std::vector<float> Data;

/** @brief Defines a convenience type to represent a shared pointer to an ordered collection of floating point data. */
typedef std::shared_ptr<Data> DataPtr;

/** @brief Defines a convenience type to represent an ordered collection of ::DataPtr objects. */
typedef std::vector<DataPtr> Example;

/** @brief Defines a convenience type to represent a shared pointer to an ordered collection of ::DataPtr objects. */
typedef std::shared_ptr<Example> ExamplePtr;

/** @brief Defines a convenience type to represent an ordered collection of ::ExamplePtr objects. */
typedef std::vector<ExamplePtr> Examples;

/**
 * @brief Defines a convenience type to represent a const_iterator over the ::ExamplePtr objects
 *           stored in an ::Examples object.
 */
typedef Examples::const_iterator ExampleIterator;

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

    /** @brief Gets an iterator pointing to the beginning of this object's ::Data */
    Data::iterator begin() {
      return data_->begin();
    }

    /** @brief Gets an iterator pointing to the end of this object's ::Data */
    Data::iterator end() {
      return data_->end();
    }

    /** @brief Gets a const iterator pointing to the beginning of this object's ::Data */
    Data::const_iterator begin() const {
      return data_->cbegin();
    }

    /** @brief Gets a const iterator pointing to the end of this object's ::Data. */
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
      return inputs_.size();
    }
};

//typedef std::shared_ptr<Batch> BatchPtr;

class DataBase {
  public:

    /** @brief Returns an iterator pointing to the beginning of this object's underlying data. */
    virtual ExampleIterator begin() const = 0;

    /** @brief Returns an iterator pointing to the end of this object's underlying data. */
    virtual ExampleIterator end() const = 0;

    /** @brief Randomly shuffles the elements of this object's underlying data. */
    virtual void shuffle() = 0;

    //virtual BatchPtr toBatch(const Examples&) = 0;

    /**
     * @brief Returns the size of the <em>i</em>-th dimension of the data.
     *
     * When an individual data point from this DataSet is used in the construction of an ExpressionGraph,
     *   the value returned by this method can be interpreted as the size of the <em>i</em>-th input to the graph.
     *
     * For example, given a DataBase of MNIST data points.
     * Each such data point contains 784 values (representing the pixel values for each of 784 pixels),
     * and a label consisting of one of 10 labels.
     * If the labels are interpreted as a one-hot vector of length 10,
     * then dim(0) would return 784,
     * and dim(1) would return 10.
     */
    virtual int dim(size_t i) {
      return (*begin())->at(i)->size();
    }
};

/**
 * @brief Convenience function to construct a new DataBase object and return a shared pointer to that object.
 *
 * The template parameters for this function specify two main pieces of information:
 * -# The first template parameter specifies the type of DataBase object to be constructed
 * -# Any subsequent template parameters specify the type(s) of any arguments to that DataBase object constructor
 *
 * @arg args An optional list of parameters which will be passed to the DataBase constructor
 *
 * @return a shared pointer to a newly constructed DataBase object
 */
template <class Set, typename ...Args>
Ptr<Set> DataSet(Args&& ...args) {
  return Ptr<Set>(new Set(std::forward<Args>(args)...));
}

}
}
