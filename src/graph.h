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

    virtual void allocate() = 0;
    
    virtual DataType val() = 0;
    virtual DataType grad() = 0;
};

typedef std::vector<Chainable<Tensor>*> ChainableStack;
typedef std::shared_ptr<Chainable<Tensor>> ChainPtr;

ChainableStack stack;

class Node : public Chainable<Tensor>,
             public keywords::Keywords {
  public:
    template <typename ...Args>
    Node(Args ...args)
     : Keywords(args...),
       shape_(Get<Shape>(keywords::shape, {1, 1})),
       name_(Get<std::string>(keywords::name, "none"))
    {
      std::cerr << "Creating node " << name_ << std::endl; 
      stack.push_back(this);
    }
    
    virtual ~Node() {};
    
    virtual void allocate() {
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
        
  protected:
    Shape shape_;
    std::string name_;
    
    Tensor val_;
    Tensor adj_;
};

}