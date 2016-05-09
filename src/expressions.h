#pragma once

namespace marian {

class Expr {
  public:
    Expr(Chainable<Tensor>* chainable) : pimpl_(chainable) {}
    
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
      
      std::cerr << "a" << std::endl;
      for(auto&& v : stack)
        v->allocate();
      
      std::cerr << "f" << std::endl;
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