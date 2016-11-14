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
#include <iostream>

#include "definitions.h"

namespace marian {

class TensorBase {
  public:
    TensorBase(float* data, Shape shape)
    : data_(data), shape_(shape)
    { }

    virtual ~TensorBase() {}

    virtual void reset(float* data) {
      data_ = data;
    }

    virtual float* data() {
      return data_;
    }

    virtual Shape& shape() {
      return shape_;
    }

    virtual size_t size() {
      return shape_.elements();
    }

    virtual float scalar() {
      UTIL_THROW_IF2(size() != 1, "Tensor is not a scalar");
      return get(0);
    }

    virtual float get(size_t) = 0;
    virtual void set(size_t, float) = 0;

    virtual void set(float) = 0;

    virtual void get(std::vector<float> &v) = 0;
    virtual void set(const std::vector<float> &v) = 0;

    virtual std::string debug() = 0;

  protected:
    float* data_;
    Shape shape_;
};

typedef std::shared_ptr<TensorBase> Tensor;

Tensor& operator<<(Tensor& t, const std::vector<float>& v);

Tensor& operator>>(Tensor& t, std::vector<float>& v);

}
