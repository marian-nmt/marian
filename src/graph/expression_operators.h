#pragma once

// This file is part of the Marian toolkit.

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

#include "graph/expression_graph.h"

namespace marian {

Expr training(Expr a);

Expr inference(Expr a);

Expr debug(Expr a, const std::string& message = "");

/**
 * @brief Associates a name with an Expr object and adds that object to the associated ExpressionGraph.
 *
 * @param a An expression object
 *
 * @return the provided Expr, after it has been named and added to the graph
 */
Expr name(Expr a, const std::string& name);

Expr rows(Expr a, const std::vector<size_t>& indeces);

Expr plus(const std::vector<Expr>&);

Expr logit(Expr a);
Expr logit(const std::vector<Expr>&);

Expr tanh(const std::vector<Expr>&);

template <typename ...Args>
Expr tanh(Args ...args) {
  std::vector<Expr> nodes{args...};
  return Expression<TanhNodeOp>(nodes);
}

Expr relu(Expr a);
Expr relu(const std::vector<Expr>&);

/**
 * Constructs a DropoutNodeOp object from the provided Expr object,
 *     wraps the <a href="https://en.wikipedia.org/wiki/Dropout_(neural_networks)">dropout</a> node in a shared pointer,
 *     adds it to the associated ExpressionGraph,
 *     and returns the shared pointer to the dropout node
 *
 * @arg a An expression object
 *
 * @see <a href="https://en.wikipedia.org/wiki/Dropout_(neural_networks)">dropout</a>
 */
//template <typename ...Args>
//Expr dropout(Expr a, Args ...args) {
//  return Expression<DropoutNodeOp>(a, args...);
//}

Expr log(Expr a);

Expr exp(Expr a);

Expr operator-(Expr a);

/*********************************************************/

Expr operator+(Expr a, Expr b);
//Expr operator+=(Expr a, Expr b);

Expr operator-(Expr a, Expr b);
//Expr operator-=(Expr a, Expr b);

Expr operator*(Expr a, Expr b);
//Expr operator*=(Expr a, Expr b);

Expr operator/(Expr a, Expr b);
//Expr operator/=(Expr a, Expr b);

Expr dot(Expr a, Expr b);

Expr transpose(Expr a);

template <typename ...Args>
Expr concatenate(const std::vector<Expr>& concats, Args ...args) {
  return Expression<ConcatenateNodeOp>(concats, args...);
}

template <typename ...Args>
Expr reshape(Expr a, Shape shape, Args ...args) {
  return Expression<ReshapeNodeOp>(a, shape, args...);
}

/*********************************************************/

template <typename ...Args>
Expr sum(Expr a, Args ...args) {
  return Expression<SumNodeOp>(a, args...);
}

Expr softmax(Expr a, Expr mask = nullptr);

Expr logsoftmax(Expr a);

template <typename ...Args>
Expr mean(Expr a, Args ...args) {
  return Expression<MeanNodeOp>(a, args...);
}

Expr cross_entropy(Expr a, Expr b);

//Expr tanh(Expr a, Expr b, Expr c);

Expr affine(Expr a, Expr b, Expr c);

template <typename ...Args>
Expr scalar_product(Expr a, Expr b, Args ...args) {
  return Expression<ScalarProductNodeOp>(a, b, args...);
}

template <typename ...Args>
Expr weighted_average(Expr in, Expr weights, Args ...args) {
  auto p = scalar_product(in, weights, args...);
  auto s = sum(weights, args...);
  return p / s;
}

Expr step(Expr a, size_t step);

Expr sqrt(Expr a, float eps = 0.f);
Expr square(Expr a);

Expr layer_norm(Expr x, Expr gamma, Expr beta = nullptr);
//Expr batch_norm(Expr x, Expr gamma, Expr beta = nullptr);

template <typename ...Args>
Expr dropout(Expr x, Args ...args) {
  auto mask = Get(keywords::mask, nullptr, args...);
  float dropout_prob = Get(keywords::dropout_prob, 0.0f, args...);

  UTIL_THROW_IF2(!mask && !dropout_prob,
                 "Neither mask nor dropout prob given");
  if(!mask) {
    auto graph = x->graph();
    mask = graph->dropout(dropout_prob, x->shape());
  }
  return x * mask;
}


}
