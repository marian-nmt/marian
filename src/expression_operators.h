#pragma once

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

#include "expression_graph.h"

namespace marian {

Expr training(Expr a);

Expr inference(Expr a);

/**
 * @brief Associates a name with an Expr object and adds that object to the associated ExpressionGraph.
 *
 * @param a An expression object
 *
 * @return the provided Expr, after it has been named and added to the graph
 */
Expr name(Expr a, const std::string& name);

Expr logit(Expr a);

Expr tanh(Expr a);

Expr relu(Expr a);

template <typename ...Args>
Expr dropout(Expr a, Args ...args) {
  return Expression<DropoutNodeOp>(a, args...);
}

Expr log(Expr a);

Expr exp(Expr a);

Expr operator-(Expr a);

/*********************************************************/

Expr operator+(Expr a, Expr b);

Expr operator-(Expr a, Expr b);

Expr operator*(Expr a, Expr b);

Expr operator/(Expr a, Expr b);

Expr dot(Expr a, Expr b);

Expr reluplus(Expr a, Expr b);


/*********************************************************/

template <typename ...Args>
Expr sum(Expr a, Args ...args) {
  return Expression<SumNodeOp>(a, args...);
}

Expr softmax(Expr a);

Expr logsoftmax(Expr a);

Expr argmax(Expr a);

template <typename ...Args>
Expr mean(Expr a, Args ...args) {
  return Expression<MeanNodeOp>(a, args...);
}

Expr cross_entropy(Expr a, Expr b);

}
