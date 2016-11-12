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

#include <set>

#include "definitions.h"
#include "tensors/tensor.h"

namespace marian {

class TensorAllocatorBase {
  public:
    virtual ~TensorAllocatorBase() {};
    virtual void reserve(size_t) = 0;
    virtual void reserveExact(size_t) = 0;
    virtual void clear() = 0;
    virtual void allocate(Tensor&, Shape) = 0;
    virtual size_t capacity() = 0;
    virtual size_t size() = 0;
    virtual Tensor asTensor() = 0;
};

template <class Device>
class TensorAllocatorDerived : public TensorAllocatorBase {
  private:
    const size_t CHUNK  = 128;
    const size_t MBYTE  = 1024 * 1024;
    const size_t FLOATS = CHUNK * MBYTE / sizeof(float);

    Device device_;
    std::vector<Tensor> allocated_;

    void reset(Tensor t, float* start) {
      t->reset(start);
    }

    void resetAllocated() {
      float* start = device_.data();
      for(auto t : allocated_) {
        reset(t, start);
        start += t->size();
      }
    }

    void checkSpace(Shape shape) {
      float* start = device_.data();
      if(!allocated_.empty()) {
        start = allocated_.back()->data() + allocated_.back()->size();
      }

      size_t available = device_.data() + device_.capacity() - start;
      if(shape.elements() > available) {
        reserve(device_.capacity() - available + shape.elements());
      }
    }

  public:
    void reserve(size_t elements = 0) {
      float mult = elements / FLOATS + 1;
      std::cerr << "Reserving " << mult * CHUNK << " MB" << std::endl;
      device_.reserve(mult * FLOATS);
      resetAllocated();
    }

    void reserveExact(size_t elements = 0) {
      size_t mbytes = (elements * sizeof(float)) / MBYTE;
      std::cerr << "Reserving space for " << elements
        << " floats (" << mbytes << " MB)" << std::endl;
      device_.reserve(elements);
      resetAllocated();
    }

    void clear() {
      allocated_.clear();
    }

    void allocate(Tensor &t, Shape shape) {
      if(!t || t->shape() != shape) {
        checkSpace(shape);

        float* start = device_.data();
        if(!allocated_.empty()) {
          start = allocated_.back()->data() + allocated_.back()->size();
        }

        t.reset(new typename Device::tensor_type(start, shape));
        allocated_.push_back(t);
      }
    }

    Tensor asTensor() {
      float* start = device_.data();
      return Tensor(new typename Device::tensor_type(start, {1, (int)size()}));
    }

    size_t capacity() {
      return device_.capacity();
    }

    size_t size() {
      float* start = device_.data();
      float* end = start;
      if(!allocated_.empty())
        end = allocated_.back()->data() + allocated_.back()->size();

      return end - start;
    }
};

typedef std::shared_ptr<TensorAllocatorBase> TensorAllocator;

template <class Device>
TensorAllocator newTensorAllocator() {
  return TensorAllocator(new TensorAllocatorDerived<Device>());
}

}
