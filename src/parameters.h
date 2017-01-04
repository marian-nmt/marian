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

#include <map>
#include <unordered_set>
#include <fstream>

#include "definitions.h"
#include "tensors/tensor_allocator.h"
#include "tensors/tensor_gpu.h"

namespace marian {

class Parameters {
  private:
    /** @brief List of all parameter nodes of this expression graph. */
    std::vector<Expr> params_;
    std::map<std::string, Expr> named_;

    TensorAllocator vals_;
    TensorAllocator grads_;

  public:
    Parameters()
      : vals_(newTensorAllocator<DeviceGPU>()),
        grads_(newTensorAllocator<DeviceGPU>())
    {}

    auto begin() -> decltype(params_.begin()) {
      return params_.begin();
    }

    auto end() -> decltype(params_.begin()) {
      return params_.end();
    }

    auto getMap() -> decltype(named_)& {
      return named_;
    }

    Expr get(const std::string& name) {
      auto it = named_.find(name);
      if(it != named_.end()) {
        return it->second;
      }
      else {
        return Expr();
      }
    }

    size_t size() {
      return params_.size();
    }

    size_t totalSize() {
      size_t sum = 0;
      for(auto p : params_)
        sum += p->shape().elements();
      return sum;
    }

    void add(Expr p, const std::string& name) {
      params_.push_back(p);
      UTIL_THROW_IF2(named_.count(name),
                     "Parameter " << name << "already exists");
      named_[name] = p;
    }

    void allocateForward() {
      if(vals_->capacity() == 0) {
        vals_->reserveExact(totalSize());
        for(auto p: params_)
          if(!p->val())
            vals_->allocate(p->val(), p->shape());
      }
    }

    void allocateBackward() {
      if(grads_->capacity() == 0) {
        grads_->reserveExact(totalSize());
        for(auto p: params_)
          if(!p->grad())
            grads_->allocate(p->grad(), p->shape());
      }
    }

    Tensor vals() {
      return vals_->asTensor();
    }

    Tensor grads() {
      return grads_->asTensor();
    }
};

}
