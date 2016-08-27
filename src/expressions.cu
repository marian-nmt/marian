#include "expressions.h"
#include "graph_operators.h"

namespace marian {

Expr::Expr(Chainable<Tensor>* chainable) : pimpl_(chainable) {}
Expr::Expr(Float v) : pimpl_(new ConstantNode(keywords::value=v,
                                              keywords::shape={1,1})) {}

Tensor Expr::val() {
  return pimpl_->val();
}

Tensor Expr::grad() {
    return pimpl_->grad();
}

ChainPtr Expr::node() {
    return pimpl_;
}
  
void Expr::forward(size_t batchSize) {
  UTIL_THROW_IF2(pimpl_.get() != Chainable<Tensor>::stack.back(),
                 "Trying to call forward on non-root of computation graph");
  
  std::cerr << "a" << std::endl;
  for(auto&& v : Chainable<Tensor>::stack) {
    v->allocate(batchSize);
  }
  
  std::cerr << "f" << std::endl;
  for(auto&& v : Chainable<Tensor>::stack)
    v->forward();    
}

void Expr::backward() {
  UTIL_THROW_IF2(pimpl_.get() != Chainable<Tensor>::stack.back(),
                "Trying to call backward on non-root of computation graph");
  
  for(auto&& v : Chainable<Tensor>::stack)
    v->set_zero_adjoint();

  typedef typename Chainable<Tensor>::ChainableStack::reverse_iterator It;
  pimpl_->init_dependent();
  for(It it = Chainable<Tensor>::stack.rbegin(); it != Chainable<Tensor>::stack.rend(); ++it)
    (*it)->backward();
}

Expr::operator ChainPtr() {
  return pimpl_;
}
    
}