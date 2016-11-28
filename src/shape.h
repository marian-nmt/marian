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
      int stride_[SHAPE_SIZE];

      /**
       * @brief Constructs a default shape.
       *
       * This default shape has four dimensions.
       * The size of each dimension is 1.
       */
      Shape()
      : shape_{1, 1, 1, 1},
        stride_{0, 0, 0, 0}
      { }

      /**
       * @brief Constructs a shape.
       *
       * @param i A list of integers representing the size of each dimension.
       */
      Shape(std::initializer_list<int> il)
      : Shape() {
        std::copy(il.begin(), il.end(), begin());
        stride_[0] = shape_[0] == 1 ? 0 : shape_[0];
        stride_[1] = shape_[1] == 1 ? 0 : 1;
        stride_[2] = shape_[2] == 1 ? 0 : shape_[0] * shape_[1];
        stride_[3] = shape_[3] == 1 ? 0 : shape_[0] * shape_[1] * shape_[2];
      }

      Shape(const Shape& shape) : Shape() {
       std::copy(shape.shape_, shape.shape_ + SHAPE_SIZE, shape_);
       std::copy(shape.stride_, shape.stride_ + SHAPE_SIZE, stride_);
      }

      /**
       * @brief Gets a reference to the int representing the size of the <code>i</code>th dimension represented by this object.
       *
       * @return a reference to the int representing the size of the <code>i</code>th dimension represented by this object
       */
      __host__ __device__
      inline int& dim(int i) {
        return shape_[i];
      }

      /**
       * @brief Gets the size of the <code>i</code>th dimension represented by this object.
       *
       * @return the size of the <code>i</code>th dimension represented by this object
       */
      __host__ __device__
      inline const int& dim(int i) const {
        return const_cast<Shape&>(*this).dim(i);
      }


      /**
       * @brief Gets a reference to the int representing the size of the <code>i</code>th dimension represented by this object.
       *
       * @return a reference to the int representing the size of the <code>i</code>th dimension represented by this object
       */
      __host__ __device__
      inline int& operator[](int i) {
        return dim(i);
      }

      /**
       * @brief Gets the size of the <code>i</code>th dimension represented by this object.
       *
       * @return the size of the <code>i</code>th dimension represented by this object
       */
      __host__ __device__
      inline const int& operator[](int i) const {
        return dim(i);
      }

      /**
       * @brief Gets a reference to the int representing the size of the <code>i</code>th dimension represented by this object.
       *
       * @return a reference to the int representing the size of the <code>i</code>th dimension represented by this object
       */
      __host__ __device__
      inline int stride(int i) {
        return stride_[i];
      }

      /**
       * @brief Gets the size of the <code>i</code>th dimension represented by this object.
       *
       * @return the size of the <code>i</code>th dimension represented by this object
       */
      __host__ __device__
      inline int stride(int i) const {
        return const_cast<Shape&>(*this).stride(i);
      }

      /**
       * @brief Gets the number of dimensions represented by this object
       *
       * @return the number of dimensions represented by this object
       */
      __host__ __device__
      inline size_t size() const {
        return SHAPE_SIZE;
      }

      /**
       * @brief Gets the total number of elements in a tensor of this shape.
       *
       * For example, if this shape represents a 5x100 tensor, this method would return 500.
       *
       * @return the total number of elements in a tensor of this shape
       */
      __host__ __device__
      inline size_t elements() const {
        return shape_[0] * shape_[1] * shape_[2] * shape_[3];
      }

      __host__ __device__
      inline int index(int i, int j, int k, int l) {
        return i * stride(0) + j * stride(1) + k * stride(2) + l * stride(3);
      }

      __host__ __device__
      inline int index(int i, Shape shape) {
        int RPV = shape[0] * shape[2] * shape[3];
        int PV  = shape[2] * shape[3];
        int V   = shape[3];

        int v = i / RPV;
        int t1 = i - v * RPV;
        int p = t1 / PV;
        int t2 = t1 - p * PV;
        int r = t2 / V;
        int c = t2 - r * V;

        return index(r, c, p, v);
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
