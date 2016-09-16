#pragma once

#include <map>

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
    ExpressionGraph() : stack_(new ChainableStack) {}
    
    void backprop(int batchSize) {
      forward(batchSize);
      backward();
    }
    
    void forward(int batchSize) {
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
    
    std::string graphviz() {
      std::stringstream ss;
      ss << "digraph ExpressionGraph {" << std::endl;
      ss << "rankdir=BT" << std::endl;
      
      typedef typename ChainableStack::reverse_iterator It;
      for(It it = stack_->rbegin(); it != stack_->rend(); ++it)
        ss << (*it)->graphviz();
      ss << "}" << std::endl;
      return ss.str();
    }
    
    /*********************************************************/
    
    template <typename ...Args>
    inline Expr input(Args ...args) {
      Expr e(this, new InputNode(args...));
      inputs_.emplace_back(e);
      return e;
    }
    
    template <typename ...Args>
    inline Expr param(Args ...args) {
      Expr e(this, new ParamNode(args...));
      params_.emplace_back(e);
      return e;
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
    
    Expr& operator[](const std::string& name) {
      auto it = named_.find(name);
      UTIL_THROW_IF2(it == named_.end(), "No such named node in graph: " << name);
      return it->second;  
    }

    bool has_node(const std::string& name) const {
      return named_.count(name) > 0;
    }
    
    void add_named_node(Expr e, const std::string& name) {
      named_.emplace(name, e);
    }
    
    std::vector<Expr>& inputs() {
      return inputs_;
    }
    
    std::vector<Expr>& params() {
      return params_;
    }
    
  private:
    ChainableStackPtr stack_;
    
    std::map<std::string, Expr> named_;
    std::vector<Expr> params_;
    std::vector<Expr> inputs_;
};

}
