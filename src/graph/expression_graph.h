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

#include <map>
#include <unordered_set>
#include <fstream>

#include "common/definitions.h"
#include "graph/chainable.h"
#include "graph/parameters.h"
#include "graph/node_operators.h"
#include "data/batch_generator.h"
#include "tensors/tensor_allocator.h"
#include "layers/param_initializers.h"
#include "3rd_party/threadpool.h"

namespace marian {

template <class T, typename ...Args>
Expr Expression(Args&& ... args);

/**
 * @brief Represents a computation graph of expressions, over which algorithmic differentiation may be performed.
 */
class ExpressionGraph : public std::enable_shared_from_this<ExpressionGraph> {
  private:

    /** @brief The full list of nodes */

    size_t count_{0};

    std::vector<Expr> nodes_;
    std::vector<std::vector<Expr>> tapes_;
    std::map<Expr, size_t> tapeMap_;

    /** @brief Maps from name to expression node. */
    std::map<std::string, Expr> named_;

    /** @brief List of all input nodes of this expression graph. */
    std::vector<Expr> inputs_;

    /** @brief Contains all nodes with regard to which we want to calculate derivatives */
    std::unordered_set<Expr> topNodes_;

    Parameters params_;
    Ptr<TensorAllocator> tensors_;

    cublasHandle_t cublasHandle_;
    size_t device_{0};

    size_t stale_{0};

  protected:
    /** @brief Constructs a new expression graph
     * Constructor is protected to force use of New<ExpressionGraph>()
    */
    ExpressionGraph() { }

    // delete copy and move constructors
    ExpressionGraph(const ExpressionGraph&) = delete;
    ExpressionGraph(ExpressionGraph&&) = delete;

    friend Ptr<ExpressionGraph> New<ExpressionGraph>();

  public:

    ~ExpressionGraph() {
      clear();
    }

    void setDevice(size_t device = 0) {
      device_ = device;
      params_.init(device);
      tensors_ = New<TensorAllocator>(device);
      cublasHandle_ = create_handle(device);
    }

    cublasHandle_t getCublasHandle() {
      return cublasHandle_;
    }

    size_t getDevice() {
      return device_;
    }

    void reserveWorkspaceMB(size_t num) {
      size_t elements = num * 1024 * 1024 / 4 - 1;
      tensors_->reserve(elements);
    }

    /**
     * @brief Performs backpropogation on this expression graph.
     *
     * Backpropogation is implemented by performing first the forward pass
     *    and then the backward pass of algorithmic differentiation (AD) on the nodes of the graph.
     *
     */
    void backprop() {
      forward();
      backward();
    }

    /**
     * @brief Perform the forward pass of algorithmic differentiation (AD) on this graph.
     *
     * This pass traverses the nodes of this graph in the order they were created;
     *    as each node is traversed, its <code>allocate()</code> method is called.
     *
     * Once allocation is complete for all nodes, this pass again traverses the nodes, in creation order;
     *    as each node is traversed, its <code>forward()</code> method is called.
     *
     * After this method has successfully completed,
     *    it is guaranteed that all node allocation has been completed,
     *    and that all forward pass computations have been performed.
     *
     * @param batchSize       XXX Marcin, could you provide a description of this param?
     */

    void forward() {
      params_.allocateForward();
      for(auto&& tape : tapes_) {
        for(auto&& v : tape) {
          v->allocate();
          v->init();
          v->forward();

          // @TODO: should be done in node
          for(auto&& child : v->children()) {
            v->decreaseEdges(1);
            child->decreaseEdges(1);
          }

          std::cerr << v->getId() << std::endl;

          if(v->marked_for_debug()) {
            std::cerr << "Debug: " << v->debug_message() << std::endl;
            std::cerr << v->val()->debug() << std::endl;
          }
        }
      }
    }

    /**
     * @brief Perform the backward pass of algorithmic differentiation (AD) on this graph.
     *
     * This pass traverses the nodes of this graph in reverse of the order they were created;
     *    as each node is traversed, its <code>set_zero_adjoint()</code> method is called.
     *
     * Once this has been performed for all nodes, this pass again traverses the nodes, again in reverse creation order;
     *    as each node is traversed, its <code>backward()</code> method is called.
     *
     * After this method has successfully completed,
     *    and that all backward pass computations have been performed.
     */
    void backward() {
      UTIL_THROW_IF2(topNodes_.size() > 1,
        "There are more than one top most node for backward step");

      params_.allocateBackward();
      params_.set_zero_adjoint();

      for(auto&& v : topNodes_)
        v->init_dependent();

      auto it = nodes_.rbegin();
      while(it != nodes_.rend()) {
        auto v = *it;

        std::cerr << v->getId() << std::endl;

        for(auto&& child: v->children())
          if(child->trainable())
            child->set_zero_adjoint();
        if(v->trainable())
          v->backward();
        for(auto&& child : v->children()) {
          v->decreaseEdges(1);
          child->decreaseEdges(1);
        }

        if(v->trainable() && v->marked_for_debug()) {
          std::cerr << "Debug Grad: " << v->debug_message() << std::endl;
          std::cerr << v->grad()->debug() << std::endl;
        }

        // delete unnamed nodes
        if(v->edges() == 0 && v->name() == "none")
          v->free();

        it++;
      }
    }

    /**
     * @brief Returns a string representing this expression graph in <code>graphviz</code> notation.
     *
     * This string can be used by <code>graphviz</code> tools to visualize the expression graph.
     *
     * @return a string representing this expression graph in <code>graphviz</code> notation
     */
    std::string graphviz() {
      std::stringstream ss;
      ss << "digraph ExpressionGraph {" << std::endl;
      //ss << "graph[splines=ortho]" << std::endl;
      ss << "rankdir=LR" << std::endl;

      auto it = nodes_.rbegin();
      while(it != nodes_.rend()) {
        auto v = *it;
        ss << v->graphviz();
        it++;
      }

      ss << "}" << std::endl;
      return ss.str();
    }

    void graphviz(const std::string& filename) {
      std::ofstream dot(filename);
      dot << graphviz();
      dot.close();
    }

    void dump(const std::string& filename) {
      std::cerr << "Saving not yet implemented" << std::endl;
    }

    /*********************************************************/

    /**
     * @brief Constructs a new node representing an input in an expression graph.
     *
     * This method records the input node in a list of input nodes,
     *    but does not attach the new input node to any existing expression graph.
     *
     * @param args           XXX Marcin, what are args here?
     *
     * @return a newly constructed input node
     */
    template <typename ...Args>
    inline Expr input(Args ...args) {
      auto e = Expression<InputNode>(shared_from_this(), args...);
      inputs_.emplace_back(e);
      return e;
    }

    /**
     * @brief Constructs a new node representing a parameter in an expression graph.
     *
     * This method records the parameter node in a list of parameter nodes,
     *    but does not attach the new parameter node to any existing expression graph.
     *
     * @param args           XXX Marcin, what are args here?
     *
     * @return a newly constructed parameter node
     */
    template <typename ...Args>
    inline Expr param(const std::string& name,
                      Shape shape,
                      Args ...args) {
      // check first if parameter already exists
      auto p = params_.get(name);
      if(p) {
        // if yes add to tape and return
        add(p);
        return p;
      }

      // if not check if name is not taken by other node
      UTIL_THROW_IF2(get(name),
                     "Non-parameter with name "
                     << name
                     << "already exists");

      // create parameter node (adds to tape)
      p = Expression<ParamNode>(shared_from_this(),
                                keywords::shape=shape,
                                args...);

      // add to list of parameters
      p->set_name(name);
      params_.add(p, name);
      return p;
    }

    /**
     * @brief Constructs a new node representing a constant in an expression graph.
     *
     * This method does not attach the new constant node to any existing expression graph.
     *
     * @param args           XXX Marcin, what are args here?
     *
     * @return a newly constructed constant node
     */
    template <typename ...Args>
    inline Expr constant(Args ...args) {
      return Expression<ConstantNode>(shared_from_this(), args...);
    }

    /**
     * @brief Constructs a new node representing a constant (with value 1) in an expression graph.
     *
     * This method does not attach the new constant node to any existing expression graph.
     *
     * @param args           XXX Marcin, what are args here?
     *
     * @return a newly constructed constant node
     */
    template <typename ...Args>
    inline Expr ones(Args ...args) {
      return Expression<ConstantNode>(shared_from_this(),
                                      keywords::init=inits::ones,
                                      args...);
    }

    /**
     * @brief Constructs a new node representing a constant (with value 0) in an expression graph.
     *
     * This method does not attach the new constant node to any existing expression graph.
     *
     * @param args           XXX Marcin, what are args here?
     *
     * @return a newly constructed constant node
     */
    template <typename ...Args>
    inline Expr zeros(Args ...args) {
      return Expression<ConstantNode>(shared_from_this(),
                                      keywords::init=inits::zeros,
                                      args...);
    }

    /*********************************************************/

    /**
     * @brief Returns the first item in the list with the specified name, if such an item exists.
     *
     * If no item with the specified name is found in the graph, this method throws an exception.
     *
     * @param name Name of the desired expression node
     *
     * @return the first item in the list with the specified name, if such an item exists
     */
    Expr get(const std::string& name) {
      auto e = params_.get(name);
      if(e)
        return e;

      auto it = named_.find(name);
      if(it == named_.end())
        return Expr();
      return it->second;
    }

    /**
     * @brief Gets the list of all parameter nodes of this expression graph
     *
     * @return the list of all parameter nodes of this expression graph
     */
    Parameters& params() {
      return params_;
    }

    /**
     * @brief Inserts an expression node with a specified name into the expression graph.
     *
     * @param e an expression node
     * @param name name of the expression node
     *
     * @return the expression node that was added to the expression graph
     */
    void add_named_node(Expr e, const std::string& name) {
      UTIL_THROW_IF2(params_.get(name) || get(name),
                     "Node names must be unique");

      named_.emplace(name, e);
    }

    void add(Expr node) {
      size_t group = 0;

      node->setId(count_++);
      for(auto& child: node->children()) {
        group = std::max(group, tapeMap_[child] + 1);
        child->increaseEdges(2);
        node->increaseEdges(2);
      }
      tapeMap_[node] = group;
      if(group >= tapes_.size())
        tapes_.resize(group + 1);
      tapes_[group].push_back(node);
      nodes_.push_back(node);
      topNodes_.insert(node);
    }

    void remove_top_node(Expr node) {
      topNodes_.erase(node);
    }

    template <class ...Args>
    void tensor(Tensor& t, Args&&... args) {
      tensors_->allocate(t, args...);
    }

    void free(Tensor& t) {
      tensors_->free(t);
    }

    void clear() {
      // clear everything apart from parameters
      count_ = 0;
      nodes_.clear();
      tapes_.clear();
      tapeMap_.clear();

      named_.clear();
      inputs_.clear();
      topNodes_.clear();
      tensors_->clear();
    }

    Expr topNode() {
      return nodes_.back();
    }
};

template <class T, typename ...Args>
Expr Expression(Args&& ... args) {
  auto e = Expr(new T(std::forward<Args>(args)...));
  e->graph()->add(e);
  return e;
}

}
