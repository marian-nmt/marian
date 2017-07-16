#pragma once

#include <fstream>
#include <map>
#include <unordered_set>

#include "3rd_party/cnpy/cnpy.h"
#include "3rd_party/threadpool.h"
#include "common/config.h"
#include "common/definitions.h"
#include "data/batch_generator.h"
#include "graph/backend.h"
#include "graph/chainable.h"
#include "graph/node_operators.h"
#include "graph/parameters.h"
#include "layers/param_initializers.h"
#include "tensors/tensor_allocator.h"

namespace marian {

template <class T, typename... Args>
Expr Expression(Args&&... args);

/**
 * @brief Represents a computation graph of expressions, over which algorithmic
 * differentiation may be performed.
 */
class ExpressionGraph : public std::enable_shared_from_this<ExpressionGraph> {
private:
  size_t count_{0};

  std::list<Expr> nodesForward_;
  std::list<Expr> nodesBackward_;

  std::unordered_set<Expr> topNodes_;
  Ptr<Parameters> params_;
  Ptr<TensorAllocator> tensors_;

  size_t device_;
  Ptr<Backend> backend_;

  std::unordered_map<size_t, WExpr> hashMap_;

  bool inferenceOnly_{false};
  bool reloaded_{false};
  std::string namespace_;

protected:
  // Delete, copy and move constructors
  ExpressionGraph(const ExpressionGraph&) = delete;
  ExpressionGraph(ExpressionGraph&&) = delete;

public:
  /** @brief Constructs a new expression graph
   *
   * Constructor should be used as New<ExpressionGraph>()
   */
  ExpressionGraph(bool inference = false);

  ~ExpressionGraph() {
    clear();
    params_->clear();
  }

  void setDevice(size_t device = 0);
  size_t getDevice() { return device_; }

  Ptr<Backend> getBackend() { return backend_; }

  void switchParams(const std::string& newNamespace) {
    namespace_ = newNamespace;
  }

  void reserveWorkspaceMB(size_t num) {
    size_t bytes = num * 1024 * 1024 - 1;
    tensors_->reserve(bytes);
  }

  void copyParams(Ptr<ExpressionGraph> graph) {
    for(auto p : *graph->params())
      param(p->name(), p->shape());
    params()->allocateForward();
    params()->vals()->copyFrom(graph->params()->vals());
  }

  void reuseWorkspace(Ptr<ExpressionGraph> graph) {
    tensors_ = graph->tensors_;
  }

  /**
   * @brief Performs backpropogation on this expression graph.
   *
   * Backpropogation is implemented by performing first the forward pass and
   * then the backward pass of algorithmic differentiation (AD) on the nodes of
   * the graph.
   */
  void backprop() {
    forward();
    backward();
  }

  bool fits() {
    try {
      tensors_->throwAtReallocation(true);
      backprop();
      tensors_->throwAtReallocation(false);
    } catch(AllocationException& e) {
      tensors_->throwAtReallocation(false);
      return false;
    }
    return true;
  }

  /**
   * @brief Perform the forward pass of algorithmic differentiation (AD) on this
   * graph.
   *
   * This pass traverses the nodes of this graph in the order they were
   * created; as each node is traversed, its <code>allocate()</code> method is
   * called.
   *
   * Once allocation is complete for all nodes, this pass again traverses the
   * nodes, in creation order; as each node is traversed, its
   * <code>forward()</code> method is called.
   *
   * After this method has successfully completed, it is guaranteed that all
   * node allocation has been completed, and that all forward pass computations
   * have been performed.
   */
  void forward() {
    params_->allocateForward();
    forwardNext();
  }

  void forwardNext() {
    // @TODO: check if allocation works properly
    hashMap_.clear();

    while(!nodesForward_.empty()) {
      auto v = nodesForward_.front();
      v->allocate();
      v->init();
      v->forward();

      if(v->marked_for_debug()) {
        std::cerr << "Debug: " << v->debug_message() << std::endl;
        std::cerr << v->val()->debug() << std::endl;
      }

      if(inferenceOnly_)
        v->children().clear();
      nodesForward_.pop_front();
    }
  }

  /**
   * @brief Perform the backward pass of algorithmic differentiation (AD) on
   * this graph.
   *
   * This pass traverses the nodes of this graph in reverse of the order they
   * were created;
   *    as each node is traversed, its <code>set_zero_adjoint()</code> method is
   * called.
   *
   * Once this has been performed for all nodes, this pass again traverses the
   * nodes, again in reverse creation order;
   *    as each node is traversed, its <code>backward()</code> method is called.
   *
   * After this method has successfully completed,
   *    and that all backward pass computations have been performed.
   */
  void backward() {
    UTIL_THROW_IF2(topNodes_.size() > 1,
                   "There are more than one top most node for backward step");

    params_->allocateBackward();
    params_->set_zero_adjoint();

    for(auto&& v : topNodes_)
      v->init_dependent();

    // named_.clear();
    topNodes_.clear();
    hashMap_.clear();

    while(!nodesBackward_.empty()) {
      auto v = nodesBackward_.back();
      nodesBackward_.pop_back();

      for(auto&& child : v->children()) {
        if(child->trainable())
          child->set_zero_adjoint();
      }
      if(v->trainable())
        v->backward();

      if(v->trainable() && v->marked_for_debug()) {
        std::cerr << "Debug Grad: " << v->debug_message() << std::endl;
        std::cerr << v->grad()->debug() << std::endl;
      }

      v->children().clear();
    }
  }

  /**
   * @brief Returns a string representing this expression graph in
   * <code>graphviz</code> notation.
   *
   * This string can be used by <code>graphviz</code> tools to visualize the
   * expression graph.
   *
   * @return a string representing this expression graph in
   * <code>graphviz</code> notation
   */
  std::string graphviz() {
    std::stringstream ss;
    ss << "digraph ExpressionGraph {" << std::endl;
    // ss << "graph[splines=ortho]" << std::endl;
    ss << "rankdir=LR" << std::endl;

    auto it = nodesForward_.rbegin();
    while(it != nodesForward_.rend()) {
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

  /*********************************************************/

  /**
   * @brief Constructs a new node representing a parameter in an expression
   * graph.
   *
   * This method records the parameter node in a list of parameter nodes, but
   * does not attach the new parameter node to any existing expression graph.
   *
   * @param args           XXX Marcin, what are args here?
   *
   * @return a newly constructed parameter node
   */
  template <typename... Args>
  Expr param(std::string name, Shape shape, Args... args) {
    if(!namespace_.empty())
      name = namespace_ + "::" + name;

    // check first if parameter already exists
    auto p = params_->get(name);
    if(p) {
      // if yes add to tape and return

      UTIL_THROW_IF2(shape != p->shape(),
                     "Requested shape for existing parameter "
                     << name
                     << " does not match original shape");

      add(p);
      return p;
    }

    // if graph was reloaded do not allow creation of new parameters
    UTIL_THROW_IF2(reloaded_,
                   "Graph was reloaded and parameter " << name << " is newly created");

    // if not check if name is not taken by other node
    UTIL_THROW_IF2(get(name),
                   "Non-parameter with name " << name << "already exists");


    // create parameter node (adds to tape)
    p = Expression<ParamNode>(
        shared_from_this(), keywords::shape = shape, args...);

    // add to list of parameters
    p->set_name(name);
    params_->add(p, name);
    return p;
  }

  /**
   * @brief Constructs a new node representing a constant in an expression
   * graph.
   *
   * This method does not attach the new constant node to any existing
   * expression graph.
   *
   * @return a newly constructed constant node
   */
  template <typename... Args>
  Expr constant(Shape shape, Args... args) {
    return Expression<ConstantNode>(
        shared_from_this(), keywords::shape = shape, args...);
  }

  /**
   * @brief Constructs a new node representing a constant (with value 1) in an
   * expression graph.
   *
   * This method does not attach the new constant node to any existing
   * expression graph.
   *
   * @param args           XXX Marcin, what are args here?
   *
   * @return a newly constructed constant node
   */
  template <typename... Args>
  Expr ones(Args... args) {
    return Expression<ConstantNode>(
        shared_from_this(), keywords::init = inits::ones, args...);
  }

  /**
   * @brief Constructs a new node representing a constant (with value 0) in an
   * expression graph.
   *
   * This method does not attach the new constant node to any existing
   * expression graph.
   *
   * @param args           XXX Marcin, what are args here?
   *
   * @return a newly constructed constant node
   */
  template <typename... Args>
  Expr zeros(Args... args) {
    return Expression<ConstantNode>(
        shared_from_this(), keywords::init = inits::zeros, args...);
  }

  Expr dropout(float prob, Shape shape);

  Expr gaussian(float mean, float stddev, Shape shape);

  /*********************************************************/

  /**
   * @brief Returns the first item in the list with the specified name, if such
   * an item exists.
   *
   * If no item with the specified name is found in the graph, this method
   * throws an exception.
   *
   * @param name Name of the desired expression node
   *
   * @return the first item in the list with the specified name, if such an item
   * exists
   */
  Expr get(std::string name) {
    if(!namespace_.empty())
      name = namespace_ + "::" + name;

    auto e = params_->get(name);
    if(e)
      return e;
    return Expr();
  }

  /**
   * @brief Gets the list of all parameter nodes of this expression graph
   *
   * @return the list of all parameter nodes of this expression graph
   */
  Ptr<Parameters>& params() { return params_; }

  Expr add(Expr node) {
    // size_t group = 0;

    size_t hash = node->hash();
    auto it = hashMap_.find(hash);
    if(it != hashMap_.end()) {
      return it->second.lock();
    }

    hashMap_[hash] = node;

    node->setId(count_++);

    nodesForward_.push_back(node);
    if(!inferenceOnly_ && node->trainable()) {
      nodesBackward_.push_back(node);
      topNodes_.insert(node);
    }

    return node;
  }

  void remove_top_node(Expr node) { topNodes_.erase(node); }

  template <class... Args>
  void tensor(Tensor& t, Args&&... args) {
    tensors_->allocate(t, args...);
  }

  void free(Tensor& t) {
    if(tensors_)
      tensors_->free(t);
  }

  void clear() {
    // clear everything apart from parameters
    count_ = 0;
    nodesForward_.clear();
    nodesBackward_.clear();

    topNodes_.clear();
    hashMap_.clear();
    tensors_->clear();
  }

  void clearParameters() { params_->clear(); }

  void setReloaded(bool reloaded) {
    reloaded_ = reloaded;
  }

  void load(const std::string& name) {
    using namespace keywords;

    LOG(info)->info("Loading model from {}", name);

    auto numpy = cnpy::npz_load(name);

    for(auto it : numpy) {
      auto name = it.first;
      // skip over special parameters starting with _
      if(name.substr(0, 8) == "special:")
        continue;

      Shape shape;
      if(it.second.shape.size() == 1) {
        shape.set(0, 1);
        shape.set(1, it.second.shape[0]);
      }
      else {
        for(int i = 0; i < it.second.shape.size(); ++i)
          shape.set(i, it.second.shape[i]);
      }

      param(name, shape, init = inits::from_numpy(it.second));
    }

    setReloaded(true);
  }

  void save(const std::string& name) {
    LOG(info)->info("Saving model to {}", name);

    std::string mode = "w";

    backend_->setDevice(getDevice());
    for(auto p : params()->getMap()) {
      std::string pName = p.first;

      if(!namespace_.empty()) {
        if(pName.substr(0, namespace_.size() + 2) == namespace_ + "::")
          pName = pName.substr(namespace_.size() + 2);
      }

      std::vector<float> v;
      p.second->val() >> v;

      unsigned shape[4];
      unsigned dim;

      auto ps = p.second->shape();
      if(ps[0] == 1 && ps[2] == 1 && ps[3] == 1) {
        shape[0] = ps[1];
        dim = 1;
        cnpy::npz_save(name, pName, v.data(), shape, dim, mode);
      } else if(ps[2] == 1 && ps[3] == 1) {
        shape[0] = ps[0];
        shape[1] = ps[1];
        dim = 2;
        cnpy::npz_save(name, pName, v.data(), shape, dim, mode);
      } else if(ps[3] == 1) {
        shape[0] = ps[0];
        shape[1] = ps[1];
        shape[2] = ps[2];
        dim = 3;
        cnpy::npz_save(name, pName, v.data(), shape, dim, mode);
      } else {
        shape[0] = ps[0];
        shape[1] = ps[1];
        shape[2] = ps[2];
        shape[3] = ps[3];
        dim = 4;
        cnpy::npz_save(name, pName, v.data(), shape, dim, mode);
      }
      mode = "a";
    }
  }
};

template <class T, typename... Args>
Expr Expression(Args&&... args) {
  // @TODO check hash, if exists do not add and return
  // cached node to minimize calculations
  auto e = Expr(new T(std::forward<Args>(args)...));
  return e->graph()->add(e);
}
}
