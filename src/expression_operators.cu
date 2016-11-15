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

#include "expression_operators.h"
#include "node_operators.h"

namespace marian {

Expr training(Expr a) {
  a->skip_inference();
  return a;
}

Expr inference(Expr a) {
  a->skip_training();
  return a;
}

Expr name(Expr a, const std::string& name) {
  a->set_name(name);
  a->graph()->add_named_node(a, name);
  return a;
}

Expr rows(Expr a, const DeviceVector<size_t>& indeces) {
  return Expression<RowsNodeOp>(a, indeces);
}

Expr logit(Expr a) {
  return Expression<LogitNodeOp>(a);
}

Expr tanh(Expr a) {
  return Expression<TanhNodeOp>(a);
}

Expr relu(Expr a) {
  return Expression<ReLUNodeOp>(a);
}

Expr log(Expr a) {
  return Expression<LogNodeOp>(a);
};

Expr exp(Expr a) {
  return Expression<ExpNodeOp>(a);
};

Expr operator-(Expr a) {
  return Expression<NegNodeOp>(a);
};

Expr softmax(Expr a) {
  return Expression<SoftmaxNodeOp>(a);
}

Expr logsoftmax(Expr a) {
  return Expression<LogSoftmaxNodeOp>(a);
}

Expr argmax(Expr a) {
  return Expression<ArgmaxNodeOp>(a);
}

/*********************************************************/

Expr operator+(Expr a, Expr b) {
  return Expression<PlusNodeOp>(a, b);
}

Expr operator-(Expr a, Expr b) {
  return Expression<MinusNodeOp>(a, b);
}

Expr operator*(Expr a, Expr b) {
  return Expression<MultNodeOp>(a, b);
}

Expr operator/(Expr a, Expr b) {
  return Expression<DivNodeOp>(a, b);
}

Expr dot(Expr a, Expr b) {
  return Expression<DotNodeOp>(a, b);
}

Expr transpose(Expr a) {
  return Expression<TransposeNodeOp>(a);
}

Expr concatenate(const std::vector<Expr>& concats) {
  return Expression<ConcatenateNodeOp>(concats);
}

Expr cross_entropy(Expr a, Expr b) {
  return Expression<CrossEntropyNodeOp>(a, b);
}

}
