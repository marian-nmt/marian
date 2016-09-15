#pragma once

#include <vector>
#include <memory>

#include "exception.h"

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
    virtual DataType &val() = 0;
    virtual DataType grad() = 0;
    virtual void setVal(DataType t) {
      UTIL_THROW2("Tensors can only be assigned to input nodes"); 
    };
};

typedef std::vector<Chainable<Tensor>*> ChainableStack;
typedef std::shared_ptr<ChainableStack> ChainableStackPtr;    
typedef std::shared_ptr<Chainable<Tensor>> ChainPtr;


}