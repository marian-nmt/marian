#pragma once

#include "keywords.h"
#include "tensor.h"

namespace marian {

template <class DataType>
struct Chainable {
    Chainable() { }
    virtual ~Chainable() { }
    virtual void forward() { }
    virtual void backward() { }
    virtual void init_dependent() { }
    virtual void set_zero_adjoint() { }

    virtual void allocate(size_t) = 0;
    
    virtual const Shape& shape() = 0;
    virtual DataType val() = 0;
    virtual DataType grad() = 0;
    virtual void setVal(DataType t) {
      UTIL_THROW2("Tensors can only be assigned to input nodes"); 
    };
    
    typedef std::vector<Chainable<DataType>*> ChainableStack;
    static ChainableStack stack;
};

template <class DataType>
typename Chainable<DataType>::ChainableStack Chainable<DataType>::stack;

typedef std::shared_ptr<Chainable<Tensor>> ChainPtr;

class Node : public Chainable<Tensor>,
             public keywords::Keywords {
  public:
    template <typename ...Args>
    Node(Args ...args)
     : Keywords(args...),
       shape_(Get<Shape>(keywords::shape, {1, 1})),
       name_(Get<std::string>(keywords::name, "none"))
    {
      stack.push_back(this);
    }
    
    virtual ~Node() {};
    
    virtual void allocate(size_t batchSize) {
      for(auto&& d : shape_) {
        if(d == whatevs)
            d = batchSize;
      }
      if(Has(keywords::lazy_shape)) {
        auto defaultShape = [this]() -> Shape { return shape_; };
        shape_ = Get<std::function<Shape()>>(keywords::lazy_shape, defaultShape)();
      }
      if(Has(keywords::lazy_value))
        val_.allocate(shape_, Get<std::function<float()>>(
          keywords::lazy_value, []()->Float{return 0.f;})());
      else if(Has(keywords::value))
        val_.allocate(shape_, Get<Float>(keywords::value, 0));
      else
        val_.allocate(shape_);
    }
    
    virtual void init_dependent() {
      if(adj_) {
        adj_.set(1);
      }
      else {
        adj_.allocate(shape_, 1);
      }
    }
    
    virtual void set_zero_adjoint() {
      if(adj_) {
        adj_.set(0);
      }
      else {
        adj_.allocate(shape_, 0);
      }
    }
    
    virtual Tensor val()  {
      UTIL_THROW_IF2(!val_, "Tensor has not been allocated");
      return val_;
    };
    
    virtual Tensor grad() {
      UTIL_THROW_IF2(!adj_, "Tensor has not been allocated");
      return adj_;
    };
    
    virtual const Shape& shape() {
      return shape_;    
    }
    
  protected:
    Shape shape_;
    std::string name_;
    
    Tensor val_;
    Tensor adj_;
};

}