#include <sstream>
#include "expression_graph.h"

namespace marian {

Expr::Expr(ExpressionGraphPtr g, Chainable<Tensor>* chainable)
  : graph_(g), pimpl_(chainable) {
  graph_->stack()->push_back(chainable);    
}

Tensor Expr::val() {
  return pimpl_->val();
}

Tensor Expr::grad() {
    return pimpl_->grad();
}

ChainPtr Expr::node() {
    return pimpl_;
}

ExpressionGraphPtr Expr::graph() {
    return graph_;
}
  
Expr::operator ChainPtr() {
  return pimpl_;
}

std::string Expr::Debug() const
{
	std::stringstream strm;
	const Shape &shape = pimpl_->shape();
	strm << marian::Debug(shape);
	return strm.str();
}

}
