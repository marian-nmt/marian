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

Expr debug(Expr a, const std::string& message) {
  a->debug(message);
  return a;
}

Expr name(Expr a, const std::string& name) {
  a->set_name(name);
  a->graph()->add_named_node(a, name);
  return a;
}

Expr rows(Expr a, const std::vector<size_t>& indeces) {
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
  auto sOrig = a->shape();
  Shape sTemp({sOrig[0] * sOrig[2] * sOrig[3], sOrig[1], 1, 1});
  return reshape(Expression<SoftmaxNodeOp>(reshape(a, sTemp)), sOrig);
}

Expr logsoftmax(Expr a) {
  auto sOrig = a->shape();
  Shape sTemp({sOrig[0] * sOrig[2] * sOrig[3], sOrig[1], 1, 1});
  return reshape(Expression<LogSoftmaxNodeOp>(reshape(a, sTemp)), sOrig);
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
  auto shapeA = a->shape();
  auto shapeB = b->shape();
  if((shapeA[2] > 1 || shapeA[3] > 1) && shapeB[2] == 1 && shapeB[3] == 1) {
    auto ra = reshape(a, {shapeA[0] * shapeA[2] * shapeA[3], shapeA[1] , 1, 1});
    return reshape(Expression<DotNodeOp>(ra, b),
                   {shapeA[0], shapeB[1], shapeA[2], shapeA[3]});
  }
  else {
    return Expression<DotNodeOp>(a, b);
  }
}

Expr transpose(Expr a) {
  return Expression<TransposeNodeOp>(a);
}

Expr cross_entropy(Expr a, Expr b) {
  auto sOrig = a->shape();
  auto sOut = a->shape();
  Shape sTemp({sOrig[0] * sOrig[2] * sOrig[3], sOrig[1], 1, 1});
  sOut.set(1, 1);
  return reshape(Expression<CrossEntropyNodeOp>(reshape(a, sTemp), b), sOut);
}

// @TODO: should be done automatically:

Expr tanhPlus3(Expr a, Expr b, Expr c) {
  std::vector<Expr> nodes = {a, b, c};
  return Expression<TanhPlus3NodeOp>(nodes);
}

Expr affine(Expr a, Expr b, Expr c) {
  auto shapeA = a->shape();
  auto shapeB = b->shape();
  if((shapeA[2] > 1 || shapeA[3] > 1) && shapeB[2] == 1 && shapeB[3] == 1) {
    auto ra = reshape(a, {shapeA[0] * shapeA[2] * shapeA[3], shapeA[1] , 1, 1});
    std::vector<Expr> nodes = {ra, b, c};
    return reshape(Expression<AffineNodeOp>(nodes),
                   {shapeA[0], shapeB[1], shapeA[2], shapeA[3]});
  }
  else {
    std::vector<Expr> nodes = {a, b, c};
    return Expression<AffineNodeOp>(nodes);
  }
}

}
