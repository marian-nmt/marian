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

#include <vector>
#include <string>
#include <functional>
#include <memory>

namespace marian {
  const size_t SHAPE_SIZE = 2;

  typedef float Float;
  
  /** @brief A placeholder that represents the size of a dimension, the actual value of which is to be specified at some later point.
   * 
   * For example, in certain cases the value of one dimension in a Shape object may be used to represent batch size.
   * In such a case, the value of batch size may not be known when the Shape object is constructed.
   * In that case, this placeholder would be used to specify that the batch size value will be defined at some later point.
   */
  const int whatevs{-1};

  /**
   * @brief Represents the size of each dimension in a tensor.
   *
   * Note: this class currently is hard-coded to 2 dimensions.
   *       This is likely to change.
   */
  class Shape {
    private:
      int shape_[SHAPE_SIZE];

    public:
    
      /**
       * @brief Constructs a default shape.
       * 
       * This default shape has two dimensions.
       * The size of each dimension is 1.
       */
      Shape() : shape_{1, 1} { }

      /**
       * @brief Constructs a shape.
       * 
       * @param i A list of integers representing the size of each dimension.
       */
      Shape(std::initializer_list<int> il) {
       std::copy(il.begin(), il.end(), begin());
      }

      /** 
       * @brief Gets a reference to the int representing the size of the <code>i</code>th dimension represented by this object.
       * 
       * @return a reference to the int representing the size of the <code>i</code>th dimension represented by this object
       */
      int& operator[](int i) {
        return shape_[i];
      }

      /** 
       * @brief Gets the size of the <code>i</code>th dimension represented by this object.
       * 
       * @return the size of the <code>i</code>th dimension represented by this object
       */
      const int& operator[](int i) const {
        return shape_[i];
      }

	  /**
	   * @brief Gets the number of dimensions represented by this object
	   * 
	   * @return the number of dimensions represented by this object
	   */ 
      size_t size() const {
        return SHAPE_SIZE;
      }

      /**
       * @brief Gets the total number of elements in a tensor of this shape.
       *
       * For example, if this shape represents a 5x100 tensor, this method would return 500.
       *
       * @return the total number of elements in a tensor of this shape
       */
      size_t totalSize() const {
        size_t s = 1;
        for(int i = 0; i < size(); ++i)
          s *= shape_[i];
        return s;
      }

      /** @brief Gets a pointer to an int that specifies the size of the first dimension represented by this object */
      int* begin() { return shape_; }

      /** @brief Gets a pointer to an int that specifies the size of the last dimension represented by this object */
      int* end() { return shape_ + SHAPE_SIZE; }

      /** @brief Gets a const pointer to an int that specifies the size of the first dimension represented by this object */
      const int* begin() const { return shape_; }
      
      /** @brief Gets a const pointer to an int that specifies the size of the last dimension represented by this object */      
      const int* end() const { return shape_+ SHAPE_SIZE; }

      /** 
       * @brief Tests this object for equality against another <code>Shape</code> object.
       *
       * @return <code>true</code> if the size of each dimension in this object
       *         is equal to the size of the corresponding dimension in the other object,
       *         <code>false</code> otherwise
       */
      bool operator==(const Shape& other) const {
        return std::equal(begin(), end(), other.begin());
      }

      /** 
       * @brief Tests this object for inequality against another <code>Shape</code> object.
       */
      bool operator!=(const Shape& other) const {
        return !(*this == other);
      }
  };
}

#include "keywords.h"

namespace marian {
  class Tensor;

  class OptimizerBase;
  typedef std::shared_ptr<OptimizerBase> OptimizerBasePtr;

  class RunBase;
  typedef std::shared_ptr<RunBase> RunBasePtr;

  // Define a set of keywords.
  //
  // Each 
  namespace keywords {
    KEY(axis, int)
    KEY(name, std::string)
    KEY(shape, Shape)
    KEY(value, float)
    KEY(lazy_shape, std::function<Shape()>)
    KEY(lazy_value, std::function<float()>)
    KEY(init, std::function<void(Tensor)>)

    KEY(optimizer, OptimizerBasePtr)
    KEY(batch_size, int)
    KEY(max_epochs, int)
    KEY(valid, RunBasePtr)
  }

}
