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

#include <cstdint>

#include "exception.h"

namespace marian {

  /**
   * @brief Represents the size of each dimension in a tensor.
   *
   * Note: this class currently is hard-coded to four dimensions.
   */

  const size_t SHAPE_SIZE = 4;


  struct Shape {
      int shape_[SHAPE_SIZE];

      /**
       * @brief Constructs a default shape.
       *
       * This default shape has four dimensions.
       * The size of each dimension is 1.
       */
      Shape() : shape_{1, 1, 1, 1} { }

      /**
       * @brief Constructs a shape.
       *
       * @param i A list of integers representing the size of each dimension.
       */
      Shape(std::initializer_list<int> il)
      : Shape() {
       std::copy(il.begin(), il.end(), begin());
      }

      Shape(const Shape& shape) : Shape() {
       std::copy(shape.begin(), shape.end(), begin());
      }

      /**
       * @brief Gets a reference to the int representing the size of the <code>i</code>th dimension represented by this object.
       *
       * @return a reference to the int representing the size of the <code>i</code>th dimension represented by this object
       */
      __host__ __device__
      int& dim(int i) {
        return shape_[i];
      }

      /**
       * @brief Gets the size of the <code>i</code>th dimension represented by this object.
       *
       * @return the size of the <code>i</code>th dimension represented by this object
       */
      __host__ __device__
      const int& dim(int i) const {
        return shape_[i];
      }


      /**
       * @brief Gets a reference to the int representing the size of the <code>i</code>th dimension represented by this object.
       *
       * @return a reference to the int representing the size of the <code>i</code>th dimension represented by this object
       */
      __host__ __device__
      int& operator[](int i) {
        return dim(i);
      }

      /**
       * @brief Gets the size of the <code>i</code>th dimension represented by this object.
       *
       * @return the size of the <code>i</code>th dimension represented by this object
       */
      __host__ __device__
      const int& operator[](int i) const {
        return dim(i);
      }

      /**
       * @brief Gets a reference to the int representing the size of the <code>i</code>th dimension represented by this object.
       *
       * @return a reference to the int representing the size of the <code>i</code>th dimension represented by this object
       */
      __host__ __device__
      int stride(int i) {
        switch(i) {
          case 0: return shape_[1];
          case 1: return 1;
          case 2: return shape_[0] * shape_[1];
          case 3: return shape_[0] * shape_[1] * shape_[2];
        }
        return 1;
      }

      /**
       * @brief Gets the size of the <code>i</code>th dimension represented by this object.
       *
       * @return the size of the <code>i</code>th dimension represented by this object
       */
      __host__ __device__
      int stride(int i) const {
        return const_cast<Shape&>(*this).stride(i);
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
      size_t elements() const {
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
