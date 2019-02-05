#pragma once

#include <fstream>
#include <map>
#include <unordered_set>

#include "common/definitions.h"
#include "graph/chainable.h"
#include "tensors/tensor_allocator.h"

namespace marian {

class Parameters {
protected:
  /** @brief List of all parameter nodes of this expression graph. */
  std::vector<Expr> params_;
  std::map<std::string, Expr> named_;

  Ptr<TensorAllocator> vals_;
  Ptr<TensorAllocator> grads_;

  size_t totalCapacity(Ptr<TensorAllocator> alloc) {
    size_t sum = 0;
    for(auto p : params_) {
      sum += alloc->capacity(p->shape(), Type::float32);
    }
    return sum;
  }

public:
  auto begin() -> decltype(params_.begin()) { return params_.begin(); }

  auto end() -> decltype(params_.begin()) { return params_.end(); }

  auto getMap() -> decltype(named_)& { return named_; }

  Expr get(const std::string& name) {
    auto it = named_.find(name);
    if(it != named_.end()) {
      return it->second;
    } else {
      return Expr();
    }
  }

  size_t size() { return params_.size(); }

  void add(Expr p, const std::string& name) {
    params_.push_back(p);
    ABORT_IF(named_.count(name), "Parameter '{}' already exists", name);
    named_[name] = p;
  }

  virtual void init(Ptr<Backend> backend) {
    vals_ = New<TensorAllocator>(backend);
    grads_ = New<TensorAllocator>(backend);
  }

  virtual void allocateForward() {
    if(!params_.empty() && vals_->size() == 0) {
      vals_->reserveExact(totalCapacity(vals_));
      for(auto p : params_) {
        if(!p->val()) {
          vals_->allocate(p->val(), p->shape());
        }
      }
    }
  }

  virtual void allocateBackward() {
    if(!params_.empty() && grads_->size() == 0) {
      grads_->reserveExact(totalCapacity(grads_));
      for(auto p : params_)
        if(!p->grad())
          grads_->allocate(p->grad(), p->shape());
    }
  }

  virtual void set_zero_adjoint() { grads()->set(0.f); }

  virtual Tensor vals() { return vals_->asTensor(); }

  virtual Tensor grads() { return grads_->asTensor(); }

  virtual void clear() {
    params_.clear();
    named_.clear();

    vals_->clear();
    grads_->clear();
  }
};

class MappedParameters : public Parameters {
private:
  Ptr<Backend> backend_;

public:
  virtual void init(Ptr<Backend> backend) override { backend_ = backend; }

  virtual void allocateForward() override {
    if(!params_.empty()) {
      for(auto p : params_) {
        if(!p->val()) {
          p->val() = Tensor(
              new TensorBase(nullptr, p->shape(), Type::float32, backend_));
        }
      }
    }
  }

  virtual void allocateBackward() override {
    ABORT("Not implemented for memory-mapped parameters");
  }

  virtual void set_zero_adjoint() override {
    ABORT("Not implemented for memory-mapped parameters");
  }

  virtual Tensor vals() override {
    ABORT("Not implemented for memory-mapped parameters");
  }

  virtual Tensor grads() override {
    ABORT("Not implemented for memory-mapped parameters");
  }

  virtual void clear() override {
    params_.clear();
    named_.clear();
  }
};

}  // namespace marian
