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

#include <vector>
#include <memory>

#include "exception.h"

/**
 * @brief Parent namespace for the Marian project
 */
namespace marian {


#define NodeOp(op) [=]() { op ; }
typedef std::vector<std::function<void()>> NodeOps;


template <class DataType>
class Chainable;
/** @brief Defines a convenience type to represent a shared pointer to a Chainable<Tensor> object. */
typedef std::shared_ptr<Chainable<Tensor>> Expr;


class ExpressionGraph;
typedef std::shared_ptr<ExpressionGraph> ExpressionGraphPtr;

/**
 * @brief Abstraction of an element in a computation graph for which a derivative can be calculated.
 *
 * The name of this class comes from the fact that
 *     this element is <a href="https://en.wikipedia.org/wiki/Function_composition">composable</a> (aka chainable)
 *     in the context of the <a href="https://en.wikipedia.org/wiki/Chain_rule">chain rule of calculus</a>.
 *
 * Given that context, in the documentation for this class, the following notation is used:
 * - Given an expression graph of composed functions,
 *   \f$y\f$ refers to the final value resulting from evaluating the entire graph
 * - \f$w_i\f$ refers to the partial result of evaluating the expression subgraph rooted at the <em>i</em>-th Chainable element
 * - \f$\bar{w}_i\f$ refers to the <a href="https://en.wikipedia.org/wiki/Automatic_differentiation#Reverse_accumulation">adjoint</a> of \f$w_i\f$,
 *   where \f$\bar{w}_i\f$ is defined as the partial derivative of \f$y\f$ with respect to \f$w_i\f$,
 *   or formally \f$\bar{w}_i = \frac{\partial y}{\partial w_i}\f$
 */
template <class DataType>
struct Chainable {
    Chainable() { }
    virtual ~Chainable() {};

    virtual void forward() = 0;
    virtual void backward() = 0;
    virtual NodeOps forwardOps() = 0;
    virtual NodeOps backwardOps() = 0;

    virtual size_t allocate() = 0;
    virtual void free() = 0;
    virtual void init() = 0;
    virtual void init_dependent() { }
    virtual void set_zero_adjoint() { }
    virtual bool trainable() = 0;
    virtual void setTrainable(bool) = 0;

    virtual void setId(size_t) = 0;
    virtual size_t getId() = 0;

    virtual void increaseEdges(size_t) = 0;
    virtual void decreaseEdges(size_t) = 0;
    virtual size_t edges() = 0;

    //virtual const std::string& type() = 0;
    virtual ExpressionGraphPtr graph() = 0;
    virtual const Shape& shape() = 0;

    virtual std::vector<Expr>& children() = 0;
    virtual DataType& val() = 0;
    virtual DataType& grad() = 0;

    virtual const std::string type() = 0;
    virtual const std::string color() = 0;
    virtual const std::string form() = 0;
    virtual const std::string label() = 0;
    virtual std::string graphviz() = 0;

    virtual void set_name(const std::string&) = 0;
    virtual const std::string &name() const = 0;

    virtual void debug(const std::string& message) = 0;
    virtual bool marked_for_debug() = 0;
    virtual const std::string& debug_message() = 0;
};

/**
 * @brief Defines a convenience type to represent an ordered collection items.
 *
 * Conceptually, the items in this collection are pointers to nodes in an expression graph.
 *
 * Naumann (2012) uses "tape" to refer to this data structure.
 * -- The Art of Differentiating Computer Programs: An Introduction to Algorithmic Differentiation, Naumann (2012)
 */
typedef std::vector<Expr> Tape;

}
