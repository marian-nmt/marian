#pragma once

#include "keywords.h"
#include "tensor.h"
#include "chainable.h"

namespace marian {

class Node : public Chainable<Tensor>,
             public keywords::Keywords {
  public:
    template <typename ...Args>
    Node(Args ...args)
     : Keywords(args...),
       shape_(Get<Shape>(keywords::shape, {1, 1})),
       name_(Get<std::string>(keywords::name, "none"))
    { }
    
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
    
    virtual Tensor &val()  {
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
