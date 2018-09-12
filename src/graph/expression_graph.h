#pragma once

#include "common/config.h"
#include "common/definitions.h"

#include "tensors/backend.h"
#include "tensors/tensor_allocator.h"

#include "graph/chainable.h"
#include "graph/node_initializers.h"
#include "graph/node_operators.h"
#include "graph/parameters.h"

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

  Tensors(Ptr<Backend> backend, Ptr<Device> device)
      : tensors_(New<TensorAllocator>(backend, device)),
        cache_(New<TensorAllocator>(backend)),
        shortterm_(New<WeakMemory>()),
        longterm_(New<Memory>()) {}

  void reserve(size_t bytes) { tensors_->reserve(bytes); }

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

  void free(Tensor& tensor) { tensors_->free(tensor); }

  // @TODO: get rid of this, not really used or can be done better
  Ptr<Allocator> allocator() { return tensors_->allocator(); }

  Expr findOrRemember(Expr node) {
    size_t hash = node->hash();
    if(node->memoize()) {
      auto it = longterm_->find(hash);
      if(it != longterm_->end()) {
        for(auto found : it->second) {
          return found;
          // @TODO: check why below code does not work for certain nodes and
          // autotuning.
          // if(node->equal(found)) {
          // std::cerr << "found memoized" << std::endl;
          // return found;
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

  void clearShorttermMemory() { shortterm_->clear(); }

  void clearLongtermMemory() { longterm_->clear(); }
};

class ExpressionGraph : public std::enable_shared_from_this<ExpressionGraph> {
private:
  size_t count_{0};

  std::list<Expr> nodesForward_;
  std::list<Expr> nodesBackward_;

  std::unordered_set<Expr> topNodes_; // current set of roots. In the end, all but one must have been consumed.

  // Holds memory and expressions that correspond to graph parameters
  Ptr<Parameters> params_;

  // Holds memory and expressions that correspond to temporary expressions.
  // This gets cleared before a new graph is built.
  Ptr<Tensors> tensors_;

  std::unordered_map<size_t, std::vector<Expr>> memoized_;

  bool inferenceOnly_{false};
  bool optimized_{false};
  Ptr<Backend> backend_;

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

  void setDevice(DeviceId deviceId = {0, DeviceType::gpu},
                 Ptr<Device> device = nullptr);

  DeviceId getDeviceId() { return backend_->getDeviceId(); }

  Ptr<Backend> getBackend() { return backend_; }

  void setOptimized(bool optimized) { optimized_ = optimized; }
  bool isOptimized() { return (optimized_ && inferenceOnly_); }

  void switchParams(const std::string& newNamespace) {
    namespace_ = newNamespace;
  }

  void copyParams(Ptr<ExpressionGraph> graph) {
    for(auto p : *graph->params())
      param(p->name(), p->shape(), inits::dummy);
    params()->allocateForward();
    params()->vals()->copyFrom(graph->params()->vals());
  }

  void reserveWorkspaceMB(size_t num) {
    size_t bytes = num * 1024 * 1024 - 1;
    tensors_->reserve(bytes);
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
    } catch(AllocationException&) {
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
        std::cerr << "Debug: " << v->debug_message() << " op=" << v->type()
                  << std::endl;
        std::cerr << v->val()->debug() << std::endl;
      }

      if(inferenceOnly_)
        v->children().clear();
      nodesForward_.pop_front();
    }
  }

  void backward(bool zero = true) {
    ABORT_IF(topNodes_.size() > 1,
             "There are more than one top most node for backward step");

    params_->allocateBackward();
    if(zero)
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
        if(child->trainable() && child->type() != "param")
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

  Ptr<Parameters>& params() { return params_; }

  Expr add(Expr node) {
    auto found = tensors_->findOrRemember(node);
    if(found) {
      return found;
    } else {
      node->setId(count_++);

      // record in foward graph
      nodesForward_.push_back(node);

      // record in backward graph if training, and keep track of roots
      if(!inferenceOnly_ && node->trainable()) {
        nodesBackward_.push_back(node);
        topNodes_.insert(node); // opportunistically record all new nodes as roots (gets removed once consumed)
      }
      for(auto child : node->children())
        topNodes_.erase(child); // this child is consumed and therefore not a root

      return node;
    }
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
  Ptr<Allocator> allocator() { return tensors_->allocator(); }

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

private:
  // convert all parameters into an array of IoItem elements, for saving
  void itemsToParameters(const std::vector<io::Item>& ioItems,
                         const std::map<std::string, std::string>& nameMap,
                         bool markReloaded = true) {
    setReloaded(false);
    for(auto& item : ioItems) {
      std::string pName = item.name;

      // skip over special parameters starting with "special:"
      if(pName.substr(0, 8) == "special:")
        continue;

      auto it = nameMap.find(pName);
      if(it != nameMap.end())
        pName = it->second;

      param(pName, item.shape, inits::from_item(item));
    }
    if(markReloaded)
      setReloaded(true);
  }

public:
  void load(const std::string& name,
            const std::map<std::string, std::string>& nameMap,
            bool markReloaded = true) {
    LOG(info, "Loading model from {}", name);
    itemsToParameters(io::loadItems(name), nameMap, markReloaded);
  }

  void load(const std::string& name, bool markReloaded = true) {
    std::map<std::string, std::string> emptyNameMap;
    load(name, emptyNameMap, markReloaded);
  }

  void load(const void* ptr,
            const std::map<std::string, std::string>& nameMap,
            bool markReloaded = true) {
    LOG(info, "Loading model from buffer at {}", ptr);
    itemsToParameters(io::loadItems(ptr), nameMap, markReloaded);
  }

  void load(const void* ptr, bool markReloaded = true) {
    std::map<std::string, std::string> emptyNameMap;
    load(ptr, emptyNameMap, markReloaded);
  }

  void mmap(const void* ptr,
            const std::map<std::string, std::string>& nameMap,
            bool markReloaded = true) {
    ABORT_IF(backend_->getDeviceId().type != DeviceType::cpu || !inferenceOnly_,
             "Memory mapping only supported for CPU inference mode");

    params_ = New<MappedParameters>();
    params_->init(backend_);

    LOG(info, "Memory mapping model at {}", ptr);
    itemsToParameters(io::mmapItems(ptr), nameMap, markReloaded);
  }

  void mmap(const void* ptr, bool markReloaded = true) {
    std::map<std::string, std::string> emptyNameMap;
    mmap(ptr, emptyNameMap, markReloaded);
  }

private:
  // convert all parameters into an array of io::Item elements, for saving
  void parametersToItems(std::vector<io::Item>& ioItems,
                         const std::map<std::string, std::string>& nameMap);

public:
  void save(const std::string& name,
            const std::string& meta,
            const std::map<std::string, std::string>& nameMap) {
    // LOG(info, "Saving model to {}", name);

    std::vector<io::Item> ioItems;
    parametersToItems(ioItems, nameMap);
    if(!meta.empty())
      io::addMetaToItems(meta, "special:model.yml", ioItems);
    io::saveItems(name, ioItems);

    // LOG(info, "Saved {} items.", ioItems.size());
  }

  void save(const std::string& name) {
    std::map<std::string, std::string> emptyNameMap;
    save(name, "", emptyNameMap);
  }

  void save(const std::string& name, const std::string& meta) {
    std::map<std::string, std::string> emptyNameMap;
    save(name, meta, emptyNameMap);
  }

  void save(const std::string& name,
            const std::map<std::string, std::string>& nameMap) {
    save(name, "", nameMap);
  }
};

template <class T, typename... Args>
Expr Expression(Args&&... args) {
  auto e = Expr(new T(std::forward<Args>(args)...));
  return e->graph()->add(e);
}
}  // namespace marian
