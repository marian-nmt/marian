#pragma once

#include "definitions.h"
#include "chainable.h"
#include "node_operators.h"
#include "tensor.h"

namespace marian {

class ExpressionGraph;
typedef ExpressionGraph* ExpressionGraphPtr;

class Expr {
  public:
    Expr(ExpressionGraphPtr g, Chainable<Tensor>* chainable);
    
    Expr operator=(Tensor t) {
      pimpl_->setVal(t);
      return *this;
    }

    Tensor val();
    Tensor grad();

    ExpressionGraphPtr graph();
    
    ChainPtr node();
    operator ChainPtr();

    std::string Debug() const;

  private:
    ExpressionGraphPtr graph_;
    ChainPtr pimpl_;
};

class ExpressionGraph {
  public:
    ExpressionGraph()
    : stack_(new ChainableStack)
    {}
    
    void forward(size_t batchSize) {
      for(auto&& v : *stack_) {
        v->allocate(batchSize);
      }
      for(auto&& v : *stack_)
        v->forward();    
    }
    
    void backward() {
      for(auto&& v : *stack_)
        v->set_zero_adjoint();
    
      typedef typename ChainableStack::reverse_iterator It;
      stack_->back()->init_dependent();
      for(It it = stack_->rbegin(); it != stack_->rend(); ++it)
        (*it)->backward();
    }
    
    template <typename ...Args>
    inline Expr input(Args ...args) {
      return Expr(this, new InputNode(args...));
    }
    
    template <typename ...Args>
    inline Expr param(Args ...args) {
      return Expr(this, new ParamNode(args...));
    }
    
    template <typename ...Args>
    inline Expr constant(Args ...args) {
      return Expr(this, new ConstantNode(args...));
    }
    
    template <typename ...Args>
    inline Expr ones(Args ...args) {
      return Expr(this, new ConstantNode(keywords::value=1, args...));
    }
    
    template <typename ...Args>
    inline Expr zeroes(Args ...args) {
      return Expr(this, new ConstantNode(keywords::value=0, args...));
    }
    
    /*********************************************************/
        
    ChainableStackPtr stack() {
      return stack_;
    }
    
  private:
    ChainableStackPtr stack_;
};

}
