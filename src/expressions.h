#pragma once

#include "definitions.h"
#include "graph.h"

namespace marian {

class Expr {
  public:
    Expr(Chainable<Tensor>* chainable);
    Expr(Float v);
    
    Expr operator=(Tensor t) {
      pimpl_->setVal(t);
      return *this;
    }
    
    Tensor val();
    Tensor grad();
    
    void forward(size_t batchSize);
    void backward();
    
    ChainPtr node();
    operator ChainPtr();
    
    std::string Debug() const;

  private:
    ChainPtr pimpl_; 
};

}
