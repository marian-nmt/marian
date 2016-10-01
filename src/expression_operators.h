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

// inefficient
template <typename ...Args>
inline Expr sum(Expr a, Args ...args) {
  using namespace keywords;
  Keywords params(args...);
  int ax = params.Get(axis, whatevs);

  if(ax == 0) {
    auto lshape = [a]() -> Shape {
      int rows = a->val().shape()[0];
      return {1, rows};
    };
    Expr one = a->graph()->ones(shape={1, a->shape()[0]},
                    lazy_shape=lshape);
    return dot(one, a);
  }
  else if(ax == 1) {
    auto lshape = [a]() -> Shape {
      int cols = a->val().shape()[1];
      //std::cerr << "Shape will be " << cols << " by 1." << std::endl;
      return {cols, 1};
    };
    Expr one = a->graph()->ones(shape={a->shape()[1], 1},
                        lazy_shape=lshape);
    return dot(a, one);
  }
  else if(ax == 2) {
    UTIL_THROW2("Not implemented");
  }
  else if(ax == 3) {
    UTIL_THROW2("Not implemented");
  }
  return sum(sum(a, axis=0), axis=1);
}

Expr softmax(Expr a);

Expr logsoftmax(Expr a);

Expr argmax(Expr a);

// inefficient
template <typename ...Args>
inline Expr mean(Expr a, Args ...args) {
  using namespace keywords;
  Keywords params(args...);
  size_t ax = params.Get(axis, whatevs);

  switch (ax) {
    case 0:
      return sum(a, axis=0) / a->graph()->constant(shape={1, 1},
                                       lazy_value=[a]() -> Float {
                                         return a->val().shape()[0];
                                       });
    case 1:
      return sum(a, axis=1) / a->graph()->constant(shape={1, 1},
                                       lazy_value=[a]() -> Float {
                                         return a->val().shape()[1];
                                       });
    case 2:
      UTIL_THROW2("Not implemented");
    case 3:
      UTIL_THROW2("Not implemented");
    default:
      return sum(a) / a->graph()->constant(shape={1, 1},
                               lazy_value=[a]() -> Float {
                                 return a->val().size();
                               });
  }
}

 Expr cross_entropy(Expr a, Expr b);

}
