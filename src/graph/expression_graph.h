#pragma once

#include "3rd_party/cnpy/cnpy.h"
#include "3rd_party/threadpool.h"
#include "common/config.h"
#include "common/definitions.h"

#include "tensors/backend.h"
#include "tensors/tensor_allocator.h"

#include "graph/chainable.h"
#include "graph/node_initializers.h"
#include "graph/node_operators.h"
#include "graph/parameters.h"

#include "3rd_party/cnpy/cnpy.h"

#include <fstream>
#include <map>
#include <unordered_set>

namespace marian {

template <class T, typename... Args>
Expr Expression(Args&&... args);

class Tensors {
private:
  Ptr<TensorAllocator> tensors_;
  Ptr<TensorAllocator> cache_;

  typedef std::unordered_map<size_t, std::vector<WExpr>> WeakMemory;
  typedef std::unordered_map<size_t, std::vector<Expr>> Memory;

  Ptr<WeakMemory> shortterm_;
  Ptr<Memory> longterm_;

public:
  Tensors(Ptr<Backend> backend)
  : tensors_(New<TensorAllocator>(backend)),
    cache_(New<TensorAllocator>(backend)),
    shortterm_(New<WeakMemory>()),
    longterm_(New<Memory>()) {}

  void reserve(size_t bytes) {
    tensors_->reserve(bytes);
  }

  void throwAtReallocation(bool throwAtRealloc) {
    tensors_->throwAtReallocation(throwAtRealloc);
  }

  void allocateForward(Expr node) {
    if(!node->val()) {
      if(node->memoize())
        cache_->allocate(node->val(), node->shape(), node->value_type());
      else
        tensors_->allocate(node->val(), node->shape(), node->value_type());
    }
  }

  void allocateBackward(Expr node) {
    if(!node->grad())
      tensors_->allocate(node->grad(), node->shape(), node->value_type());
  }

  void free(Tensor& tensor) {
    tensors_->free(tensor);
  }

  // @TODO: get rid of this, not really used or can be done better
  Ptr<Allocator> allocator() {
    return tensors_->allocator();
  }

  Expr findOrRemember(Expr node) {
    size_t hash = node->hash();
    if(node->memoize()) {
      auto it = longterm_->find(hash);
      if(it != longterm_->end()) {
        for(auto found : it->second) {
          return found;
          // @TODO: check why below code does not work for certain nodes and autotuning.
          //if(node->equal(found)) {
            //std::cerr << "found memoized" << std::endl;
            //return found;
          //}
        }
      }
      (*longterm_)[hash].push_back(node);
    }

    auto it = shortterm_->find(hash);
    if(it != shortterm_->end()) {
      for(auto foundWeak : it->second) {
        auto found = foundWeak.lock();
        if(node->equal(found)) {
          return found;
        }
      }
    }
    (*shortterm_)[hash].push_back(node);
    return nullptr;
  }

  void clear() {
    tensors_->clear();
    shortterm_->clear();
  }

  void clearShorttermMemory() {
    shortterm_->clear();
  }

  void clearLongtermMemory() {
    longterm_->clear();
  }

};

class ExpressionGraph : public std::enable_shared_from_this<ExpressionGraph> {
private:
  size_t count_{0};

  std::list<Expr> nodesForward_;
  std::list<Expr> nodesBackward_;

  std::unordered_set<Expr> topNodes_;
  Ptr<Parameters> params_;
  Ptr<Tensors> tensors_;

  Ptr<Backend> backend_;

  std::unordered_map<size_t, std::vector<Expr>> memoized_;

  bool inferenceOnly_{false};
  bool optimized_{false};

  bool reloaded_{false};
  std::string namespace_;

  bool throwNaN_{false};

protected:
  // Delete, copy and move constructors
  ExpressionGraph(const ExpressionGraph&) = delete;
  ExpressionGraph(ExpressionGraph&&) = delete;

public:
  /** @brief Constructs a new expression graph
   *
   * Constructor should be used as New<ExpressionGraph>()
   */
  ExpressionGraph(bool inference = false, bool optimized = false);

  void setInference(bool inference) { inferenceOnly_ = inference; }
  bool isInference() { return inferenceOnly_; }

  ~ExpressionGraph() {
    clear();
    params_->clear();
  }

  void setDevice(DeviceId deviceId = {0, DeviceType::gpu});

  DeviceId getDevice() { return backend_->getDevice(); }

  Ptr<Backend> getBackend() { return backend_; }

  void setOptimized(bool optimized) { optimized_ = optimized; }
  bool isOptimized() { return (optimized_ && inferenceOnly_); }

  void switchParams(const std::string& newNamespace) {
    namespace_ = newNamespace;
  }

  void reserveWorkspaceMB(size_t num) {
    size_t bytes = num * 1024 * 1024 - 1;
    tensors_->reserve(bytes);
  }

  void copyParams(Ptr<ExpressionGraph> graph) {
    for(auto p : *graph->params())
      param(p->name(), p->shape(), inits::dummy);
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

  void forward() {
    params_->allocateForward();
    forwardNext();
  }

  void checkNan(Tensor t);

  void forwardNext() {
    // @TODO: check if allocation works properly
    tensors_->clearShorttermMemory();

    while(!nodesForward_.empty()) {
      auto v = nodesForward_.front();
      v->allocate();
      v->init();
      v->forward();

      checkNan(v->val());

      if(v->marked_for_debug()) {
        std::cerr << "Debug: " << v->debug_message() << " op=" << v->type() << std::endl;
        std::cerr << v->val()->debug() << std::endl;
      }

      if(inferenceOnly_)
        v->children().clear();
      nodesForward_.pop_front();
    }
  }

  void backward() {
    ABORT_IF(topNodes_.size() > 1,
             "There are more than one top most node for backward step");

    params_->allocateBackward();
    params_->set_zero_adjoint();

    for(auto&& v : topNodes_)
      v->init_dependent();

    // named_.clear();
    topNodes_.clear();

    tensors_->clearShorttermMemory();

    while(!nodesBackward_.empty()) {
      auto v = nodesBackward_.back();
      nodesBackward_.pop_back();

      for(auto&& child : v->children()) {
        if(child->trainable())
          child->set_zero_adjoint();
      }

      if(v->trainable())
        v->backward();

      checkNan(v->grad());

      if(v->trainable() && v->marked_for_debug()) {
        std::cerr << "Debug Grad: " << v->debug_message() << std::endl;
        std::cerr << v->grad()->debug() << std::endl;
      }

      v->children().clear();
    }
  }

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

  Expr param(const std::string& pname,
             const Shape& shape,
             const NodeInitializer& init,
             bool fixed = false) {
    std::string name = pname;
    if(!namespace_.empty())
      name = namespace_ + "::" + name;

    // check first if parameter already exists
    auto p = params_->get(name);
    if(p) {
      // if yes add to tape and return
      ABORT_IF(shape != p->shape(),
               "Requested shape {} for existing parameter '{}' does not match "
               "original shape {}",
               shape,
               name,
               p->shape());

      p->setTrainable(!fixed);
      add(p);
      return p;
    }

    // if graph was reloaded do not allow creation of new parameters
    ABORT_IF(reloaded_,
             "Graph was reloaded and parameter '{}' is newly created",
             name);

    // if not check if name is not taken by other node
    ABORT_IF(get(name), "Non-parameter with name '{}' already exists", name);

    // create parameter node (adds to tape)
    p = Expression<ParamNode>(shared_from_this(), shape, init, fixed);

    // add to list of parameters
    p->set_name(name);
    params_->add(p, name);
    return p;
  }

  Expr constant(const Shape& shape, const NodeInitializer& init) {
    return Expression<ConstantNode>(shared_from_this(), shape, init);
  }

  Expr ones(const Shape& shape) {
    return Expression<ConstantNode>(shared_from_this(), shape, inits::ones);
  }

  Expr zeros(const Shape& shape) {
    return Expression<ConstantNode>(shared_from_this(), shape, inits::zeros);
  }

  Expr dropout(float prob, const Shape& shape);

  Expr get(std::string name) {
    if(!namespace_.empty())
      name = namespace_ + "::" + name;

    auto e = params_->get(name);
    if(e)
      return e;
    return Expr();
  }

  Ptr<Parameters>& params() {
    return params_;
  }

  Expr add(Expr node) {

    auto found = tensors_->findOrRemember(node);
    if(found) {
      return found;
    } else {
      node->setId(count_++);

      nodesForward_.push_back(node);
      if(!inferenceOnly_ && node->trainable()) {
        nodesBackward_.push_back(node);
        topNodes_.insert(node);
      }

      return node;
    }
  }

  void remove_top_node(Expr node) {
    topNodes_.erase(node);
  }

  void allocateForward(Expr node) {
    if(tensors_)
      tensors_->allocateForward(node);
  }

  void allocateBackward(Expr node) {
    if(tensors_)
      tensors_->allocateBackward(node);
  }

  void free(Tensor& tensor) {
    if(tensors_)
      tensors_->free(tensor);
  }

  // @TODO: get rid of this, not really used or can be done better
  Ptr<Allocator> allocator() {
    return tensors_->allocator();
  }

  void clear() {
    // clear everything apart from parameters and memoized nodes
    count_ = 0;
    nodesForward_.clear();
    nodesBackward_.clear();

    topNodes_.clear();

    tensors_->clear();
  }

  void clearParameters() { params_->clear(); }

  void setReloaded(bool reloaded) { reloaded_ = reloaded; }

  void setThrowNaN(bool throwNaN) { throwNaN_ = throwNaN; }

  void load(const std::string& name, bool markReloaded) {
    using namespace keywords;

    LOG(info, "Loading model from {}", name);
    setReloaded(false);

    auto numpy = cnpy::npz_load(name);

    for(auto it : numpy) {
      auto name = it.first;
      // skip over special parameters starting with _
      if(name.substr(0, 8) == "special:")
        continue;

      Shape shape;
      if(it.second->shape.size() == 1) {
        shape.resize(2);
        shape.set(0, 1);
        shape.set(1, it.second->shape[0]);
      } else {
        shape.resize(it.second->shape.size());
        for(int i = 0; i < it.second->shape.size(); ++i)
          shape.set(i, it.second->shape[i]);
      }

      param(name, shape, inits::from_numpy(it.second));
    }

    if(markReloaded)
      setReloaded(true);
  }

  // convert all parameters into an array pf cnpy::NpzItem elements, for saving
  void save(std::vector<cnpy::NpzItem>& npzItems)
  {
    for(auto p : params()->getMap()) {
      std::string pName = p.first;

      if(!namespace_.empty()) {
        if(pName.substr(0, namespace_.size() + 2) == namespace_ + "::")
          pName = pName.substr(namespace_.size() + 2);
      }

      std::vector<float> v;
      p.second->val()->get(v);

      auto& pShape = p.second->shape();
      std::vector<unsigned int> shape(pShape.begin(), pShape.end());

      npzItems.emplace_back(pName, v, shape);
    }
  }

  void save(const std::string& name) {
    LOG(info, "Saving model to {}", name);
    std::vector<cnpy::NpzItem> npzItems;
    save(npzItems);
    cnpy::npz_save(name, npzItems);
    LOG(info, "Saved {} items.", npzItems.size());
  }
};

template <class T, typename... Args>
Expr Expression(Args&&... args) {
  auto e = Expr(new T(std::forward<Args>(args)...));
  return e->graph()->add(e);
}
}
