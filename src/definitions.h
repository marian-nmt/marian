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

#define SHAPE_SIZE 2

namespace marian {
  typedef float Float;
  const int whatevs{-1};
  
  // POD for shape
  class Shape {
    private:
      int shape_[SHAPE_SIZE];
      
    public:
      Shape() : shape_{1, 1} { }
      
      Shape(std::initializer_list<int> il) {
       std::copy(il.begin(), il.end(), begin());
      }
    
      int& operator[](int i) {
        return shape_[i];
      }
      
      const int& operator[](int i) const {
        return shape_[i];
      }
      
      size_t size() const {
        return SHAPE_SIZE;
      }
      
      int* begin() { return shape_; }
      int* end() { return shape_ + SHAPE_SIZE; }

      const int* begin() const { return shape_; }
      const int* end() const { return shape_+ SHAPE_SIZE; }
  };
}

#include "keywords.h"

namespace marian {
  class Tensor;

  namespace keywords {
    KEY(axis, int)
    KEY(name, std::string)
    KEY(shape, Shape)
    KEY(value, float)
    KEY(lazy_shape, std::function<Shape()>)
    KEY(lazy_value, std::function<float()>)
    KEY(init, std::function<void(Tensor)>)
  }

}
