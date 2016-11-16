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

#include "definitions.h"
#include "chainable.h"
#include "parameters.h"
#include "node_operators.h"
#include "batch_generator.h"
#include "tensors/tensor_allocator.h"
#include "tensors/tensor_gpu.h"

namespace marian {

template <class T, typename ...Args>
Expr Expression(Args&& ... args);

/**
 * @brief Represents a computation graph of expressions, over which algorithmic differentiation may be performed.
 */
class ExpressionGraph : public std::enable_shared_from_this<ExpressionGraph> {
  private:
    /** @brief Constructs a new expression graph
     * Constructor is private to force use of New<ExpressionGraph>()
    */
    ExpressionGraph() : tensors_(newTensorAllocator<DeviceGPU>()) {}

    // delete copy and move constructors
    ExpressionGraph(const ExpressionGraph&) = delete;
    ExpressionGraph(ExpressionGraph&&) = delete;

    friend ExpressionGraphPtr New<ExpressionGraph>();

  public:

    void setInputs(data::BatchPtr batch) {
      auto& bInputs = batch->inputs();
      auto& gInputs = this->inputs();

      UTIL_THROW_IF2(bInputs.size() != gInputs.size(),
                     "Number of batch inputs does not correspond to number of input nodes");

      for(int i = 0; i < gInputs.size(); ++i) {
        if(!gInputs[i]->val())
          tensor(gInputs[i]->val(), bInputs[i].shape());
        gInputs[i]->val()->set(bInputs[i].data());
      }
    }

    /**
     * @brief Performs backpropogation on this expression graph.
     *
     * Backpropogation is implemented by performing first the forward pass
     *    and then the backward pass of algorithmic differentiation (AD) on the nodes of the graph.
     *
     * @param batch A batch of training data
     */
    void backprop(data::BatchPtr batch) {
      forward(batch);
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
    void forward(data::BatchPtr batch) {
      params_.allocateForward();

      for(auto&& v : tape_)
        if(!v->skipped_training())
          v->allocate(batch->dim());

      setInputs(batch);

      for(auto&& v : tape_)
        if(!v->skipped_training()) {
          v->forward();
          if(v->marked_for_debug()) {
            std::cerr << "Debug: " << v->debug_message() << std::endl;
            std::cerr << v->val()->debug() << std::endl;
          }
        }
    }

    void forward() {
      params_.allocateForward();

      for(auto&& v : tape_)
        if(!v->skipped_training())
          v->allocate(0);

      for(auto&& v : tape_)
        if(!v->skipped_training()) {
          v->forward();
          if(v->marked_for_debug()) {
            std::cerr << "Debug: " << v->debug_message() << std::endl;
            std::cerr << v->val()->debug() << std::endl;
          }
        }
    }

    void inference(data::BatchPtr batch) {
      for(auto&& v : tape_)
        if(!v->skipped_inference())
          v->allocate(batch->dim());

      // @TODO create setInputsInference !
      setInputs(batch);

      for(auto&& v : tape_)
        if(!v->skipped_inference()) {
          v->inference();
          if(v->marked_for_debug()) {
            std::cerr << "Debug: " << v->debug_message() << std::endl;
            std::cerr << v->val()->debug() << std::endl;
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
     *https://www.facebook.com/
     * After this method has successfully completed,
     *    and that all backward pass computations have been performed.
     */
    void backward() {
      UTIL_THROW_IF2(topNodes_.size() > 1,
        "There are more than one top most node for backward step");

      params_.allocateBackward();

      for(auto&& v : tape_)
        if(!v->skipped_training())
          v->set_zero_adjoint();

      typedef typename Tape::reverse_iterator It;
      It it = tape_.rbegin();
      while(topNodes_.count(*it) == 0 && it != tape_.rend())
        it++;
      (*it)->init_dependent();
      while(it != tape_.rend()) {
        if(!(*it)->skipped_training())
          (*it)->backward();
        it++;
      }
    }

    void backward_debug(Float delta) {
      UTIL_THROW_IF2(topNodes_.size() > 1,
        "There are more than one top most node for backward step");

      for(auto&& v : tape_)
        if(!v->skipped_training())
          v->set_zero_adjoint();

      typedef typename Tape::reverse_iterator It;
      It it = tape_.rbegin();
      while(topNodes_.count(*it) == 0 && it != tape_.rend())
        it++;
      (*it)->init_dependent();
      while(it != tape_.rend()) {
        if(!(*it)->skipped_training())
          (*it)->backward_debug(delta);
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
      ss << "graph[splines=ortho]";
      ss << "rankdir=LR" << std::endl;
      typedef typename Tape::reverse_iterator It;
      for(It it = tape_.rbegin(); it != tape_.rend(); ++it) {
        ss << (*it)->graphviz();
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
      return Expression<ConstantNode>(shared_from_this(), keywords::value=1, args...);
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
      return Expression<ConstantNode>(shared_from_this(), keywords::value=0, args...);
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
     * @brief Gets the list of all input nodes of this expression graph
     *
     * @return the list of all input nodes of this expression graph
     */
    std::vector<Expr>& inputs() {
      return inputs_;
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
      tape_.push_back(node);
      if(!node->skipped_training())
        topNodes_.insert(node);
    }

    void remove_top_node(Expr node) {
      topNodes_.erase(node);
    }

    template <class ...Args>
    void tensor(Tensor& t, Args&&... args) {
      tensors_->allocate(t, args...);
    }

    void clear() {
      // clear everything apart from parameters
      tape_.clear();
      named_.clear();
      inputs_.clear();
      topNodes_.clear();
      tensors_->clear();
    }

  private:

    /** @brief The full list of nodes */
    Tape tape_;

    /** @brief Maps from name to expression node. */
    std::map<std::string, Expr> named_;

    /** @brief List of all input nodes of this expression graph. */
    std::vector<Expr> inputs_;

    /** @brief Contains all nodes with regard to which we want to calculate derivatives */
    std::unordered_set<Expr> topNodes_;

    Parameters params_;
    TensorAllocator tensors_;
};

/** @brief A pointer to an expression graph. */
typedef std::shared_ptr<ExpressionGraph> ExpressionGraphPtr;

template <class T, typename ...Args>
Expr Expression(Args&& ... args) {
  auto e = Expr(new T(std::forward<Args>(args)...));
  e->graph()->add(e);
  return e;
}

}
