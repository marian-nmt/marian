#pragma once

#include <memory>
#include <functional>
#include <vector>
#include <cmath>

namespace marian {

class TensorImpl {
  public:
    typedef float value_type;
    
    TensorImpl(size_t size, value_type value)
    : data_(size, value), tno_(tensorCounter++)
    {
      std::cerr << "Allocating tensor " << tno_ << std::endl;
    }
   
   TensorImpl(const TensorImpl& t)
    : data_(t.data_.begin(), t.data_.end())
    {
      std::cerr << "Copying tensor " << tno_ << std::endl;
    }
    
    ~TensorImpl() {
      std::cerr << "Destroying tensor " << tno_ << std::endl;
    }
   
    size_t size() const {
      return data_.size();
    }
    
    value_type* data() {
      return data_.data();
    }
    
    const value_type* data() const {
      return data_.data();
    }
    
    size_t id() const {
      return tno_;
    }
    
    void set(value_type value) {
      std::fill(data_.begin(), data_.end(), value);
    }
    
  private:
    std::vector<value_type> data_;
    size_t tno_;
    
    static size_t tensorCounter;
};

size_t TensorImpl::tensorCounter = 0;

class Tensor {
  public:
    typedef TensorImpl::value_type value_type;
    
    Tensor(size_t size, float value)
      : pimpl_(new TensorImpl(size, value)) {}
    
    Tensor() {}
    
    ~Tensor() {}
    
    size_t size() const {
      return pimpl_->size();
    }
    
    float* data() {
      return pimpl_->data();
    }
    
    const float* data() const {
      return pimpl_->data();
    }
    
    void set(float value) {
      pimpl_->set(value);
    }
    
    size_t id() const {
      return pimpl_->id();
    }
    
  private:
    std::shared_ptr<TensorImpl> pimpl_;
};

Tensor operator+(const Tensor a, const Tensor b) {
  Tensor c(a.size(), 0);
  for(size_t i = 0; i < a.size(); ++i) {
    c.data()[i] = a.data()[i] + b.data()[i];
  }
  return c;
}

Tensor operator*(const Tensor a, const Tensor b) {
  Tensor c(a.size(), 0);
  for(size_t i = 0; i < a.size(); ++i) {
    c.data()[i] = a.data()[i] * b.data()[i];
  }
  return c;
}

Tensor operator+=(Tensor a, const Tensor b) {
  for(size_t i = 0; i < a.size(); ++i) {
    a.data()[i] += b.data()[i];
  }
  return a;
}

}