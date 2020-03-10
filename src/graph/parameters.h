#pragma once

#include <fstream>
#include <map>
#include <unordered_set>

#include "common/definitions.h"
#include "graph/chainable.h"
#include "tensors/tensor_allocator.h"

namespace marian {

// @TODO: Currently an ExpressionGraph only supports one Parameters object and
// the type of parameters has to be the inside on Parameters object. This limits
// parameter types to a single chosen type, e.g. only fp32 or only fp16. This should
// be extended to allow multiple sets of parameters.
// The reason here is to be able to efficiently compute updates of whole parameter
// sets of one type.
class Parameters {
protected:
  Type acceptedElementType_; // this parameter object only takes paramters of this type

  /** @brief List of all parameter nodes of this expression graph. */
  std::vector<Expr> params_;
  std::map<std::string, Expr> named_;

  Ptr<TensorAllocator> vals_;
  Ptr<TensorAllocator> grads_;

  size_t totalCapacity(Ptr<TensorAllocator> alloc) {
    size_t sum = 0;
    for(auto p : params_) {
      sum += alloc->capacity(p->shape(), p->value_type());
    }
    return sum;
  }

public:
  Parameters(Type acceptedType) : acceptedElementType_(acceptedType) {
    LOG(debug, "Created parameter object of type {}", acceptedElementType_);
  }

  virtual ~Parameters() {
    LOG(debug, "Destroyed parameter object of type {}", acceptedElementType_);
  }

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
    LOG(debug, "Adding parameter {} to parameter object of type {}", name, acceptedElementType_);

    ABORT_IF(named_.count(name), "Parameter '{}' already exists", name);
    ABORT_IF(p->value_type() != acceptedElementType_,
             "Requested parameter type ({}) is different from chosen parameter type ({})",
             p->value_type(), acceptedElementType_);
    params_.push_back(p);
    named_[name] = p;
  }

  virtual void init(Ptr<Backend> backend) {
    vals_ = New<TensorAllocator>(backend);
    grads_ = New<TensorAllocator>(backend);
  }

  virtual void init(Ptr<Backend> backend, Ptr<Device> device) {
    vals_ = New<TensorAllocator>(backend, device);
    grads_ = New<TensorAllocator>(backend, device);
  }

  virtual void allocateForward() {
    if(!params_.empty() && vals_->size() == 0) {
      vals_->reserveExact(totalCapacity(vals_));

      // sort parameters by name before allocation to make sure the memory layout after allocation is always the same
      std::sort(params_.begin(), params_.end(), [](Expr n1, Expr n2){ return n1->name() < n2->name(); });

      for(auto p : params_) {
        if(!p->val()) {
          vals_->allocate(p->val(), p->shape(), p->value_type());
        }
      }
    }
  }

  virtual void allocateBackward() {
    if(!params_.empty() && grads_->size() == 0) {

      // sort parameters by name before allocation to make sure the memory layout after allocation is always the same
      std::sort(params_.begin(), params_.end(), [](Expr n1, Expr n2){ return n1->name() < n2->name(); });

      grads_->reserveExact(totalCapacity(grads_));
      for(auto p : params_)
        if(!p->grad())
          grads_->allocate(p->grad(), p->shape(), p->value_type());
    }
  }

  virtual void set_zero_adjoint() { grads()->set(0.f); }

  virtual Tensor vals() { return vals_->asTensor(acceptedElementType_); }

  virtual Tensor grads() { return grads_->asTensor(acceptedElementType_); }

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
  MappedParameters(Type acceptedElementType) : Parameters(acceptedElementType) {
    LOG(debug, "Created mapped parameter object of type {}", acceptedElementType);
  }

  virtual void init(Ptr<Backend> backend) override { backend_ = backend; }
  virtual void init(Ptr<Backend> backend, Ptr<Device>) override { init(backend); }

  virtual void allocateForward() override {
    if(!params_.empty()) {
      for(auto p : params_) {
        if(!p->val()) {
          p->val() = TensorBase::New(nullptr, p->shape(), p->value_type(), backend_);
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
