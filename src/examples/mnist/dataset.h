#pragma once

#include <memory>
#include <vector>

#include "common/definitions.h"


namespace marian {
namespace data {

typedef std::vector<float> Data;

typedef std::shared_ptr<Data> DataPtr;
typedef std::vector<DataPtr> Example;

typedef std::shared_ptr<Example> ExamplePtr;
typedef std::vector<ExamplePtr> Examples;

typedef Examples::const_iterator ExampleIterator;


class DataBase {
  public:
    virtual ExampleIterator begin() const = 0;
    virtual ExampleIterator end() const = 0;
    virtual void shuffle() = 0;

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

template <class Set, typename ...Args>
Ptr<Set> DataSet(Args&& ...args) {
  return Ptr<Set>(new Set(std::forward<Args>(args)...));
}

}
}
