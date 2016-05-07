#pragma once

#include <memory>
#include <functional>
#include <vector>
#include <cmath>

#include "exception.h"
#include "cudnn_tensor.h"

namespace marian {

template <class DataType>
struct Chainable : public std::enable_shared_from_this<Chainable<DataType>> {
    Chainable() { }
    virtual ~Chainable() { }
    virtual void forward() { }
    virtual void backward() { }
    virtual void init_dependent() { }
    virtual void set_zero_adjoint() { }

    virtual DataType val() = 0;
    virtual DataType grad() = 0;
};

typedef std::vector<Chainable<Tensor>*> ChainableStack;
typedef std::shared_ptr<Chainable<Tensor>> ChainPtr;

ChainableStack stack;

class Node : public Chainable<Tensor> {
  public:
    Node(const Tensor t) : val_(t) {
      //std::cerr << "Putting node with tensor " << t.id() << " on stack" << std::endl;
      stack.push_back(this);
    }
    
    virtual ~Node() {};
    
    virtual void init_dependent() {
      if(adj_) {
        adj_.set(1);
      }
      else {
        adj_ = Tensor(val_.shape(), 1);
      }
    }
    
    virtual void set_zero_adjoint() {
      if(adj_) {
        adj_.set(0);
      }
      else {
        adj_ = Tensor(val_.shape(), 0);
      }
    }
    
    virtual Tensor val()  { return val_; };
    virtual Tensor grad() { return adj_; };
        
  protected:
    Tensor val_;
    Tensor adj_;
};

class Var {
  public:
    Var() : pimpl_(nullptr) {}
    Var(const Tensor t) : pimpl_(new Node(t)) {}
    Var(const Tensor::value_type v) : pimpl_(new Node(Tensor(v))) {}
    Var(const ChainPtr chainable) : pimpl_(chainable) {}
    Var(Chainable<Tensor>* chainable) : pimpl_(chainable) {}
    
    Tensor val() {
      return pimpl_->val();
    }
    
    Tensor grad() {
        return pimpl_->grad();
    }
    
    ChainPtr pimpl() {
        return pimpl_;
    }
    
    void forward() {
      UTIL_THROW_IF2(pimpl_.get() != stack.back(),
                     "Trying to call forward on non-root of computation graph");
      
      for(auto&& v : stack)
        v->forward();    
    }
    
    void backward() {
      UTIL_THROW_IF2(pimpl_.get() != stack.back(),
                     "Trying to call backward on non-root of computation graph");
      
      for(auto&& v : stack)
        v->set_zero_adjoint();
    
      typedef ChainableStack::reverse_iterator It;
      pimpl_->init_dependent();
      for(It it = stack.rbegin(); it != stack.rend(); ++it)
        (*it)->backward();
    }
    
    operator ChainPtr() {
      return pimpl_;
    }
    
  private:
    ChainPtr pimpl_; 
};

}