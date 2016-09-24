// This file is part of the Marian toolkit.
// Marian is copyright (c) 2016 Marcin Junczys-Dowmunt.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

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

void Expr::setVal(const Tensor &val) {
  pimpl_->setVal(val);
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
