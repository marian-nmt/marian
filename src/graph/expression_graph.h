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

  void free(const Tensor& tensor) { tensors_->free(tensor); }

  // @TODO: get rid of this, not really used or can be done better
  Ptr<Allocator> allocator() { return tensors_->allocator(); }

  Expr findOrRemember(Expr node) {
    size_t hash = node->hash();
    // memoize constant nodes that are not parameters
    // parameters are already memoized in the graph itself
    if(node->type() != "param" && node->memoize()) {
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
      for(auto found : it->second) {
        if(node->equal(found)) {
          return found;
        }
      }
    }
    (*shortterm_)[hash].push_back(node.get()); // weakPtr
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
  size_t count_{0};

  std::list<Expr> nodesForward_;
  std::list<Expr> nodesBackward_;

  std::unordered_set<Expr> topNodes_; // current set of roots. In the end, all but one must have been consumed.

  // Holds memory and expressions that correspond to temporary expressions.
  // This gets cleared before a new graph is built.
  Ptr<Tensors> tensors_;

  std::unordered_map<size_t, std::vector<Expr>> memoized_;

  Type parameterType_{Type::float32}; // Type used for storing parameters, currently all parameters have to have the same type
  Type saveType_{Type::float32};      // Type used for saving to disk, can be different, e.g. double or float16.

  bool inferenceOnly_{false};

  bool optimized_{false};     // during inference, use optimizations that might lead to precision loss, e.g. 8-bit MatMul.
  bool checkpointing_{false}; // use gradient checkpointing if true

  bool reloaded_{false};
  std::string namespace_;

  bool throwNaN_{false};

protected:
  // Delete, copy and move constructors
  ExpressionGraph(const ExpressionGraph&) = delete;
  ExpressionGraph(ExpressionGraph&&) = delete;

  // Holds memory and expressions that correspond to graph parameters
  Ptr<Parameters> params_;
  Ptr<Backend> backend_;

public:
  /** @brief Constructs a new expression graph
   *
   * Constructor should be used as New<ExpressionGraph>()
   */
  ExpressionGraph(bool inference = false);

  virtual ~ExpressionGraph() {
    clear();
    params_->clear();
  }

  virtual void setDevice(DeviceId deviceId = {0, DeviceType::gpu},
                 Ptr<Device> device = nullptr);

  DeviceId getDeviceId() { return backend_->getDeviceId(); }

  Ptr<Backend> getBackend() { return backend_; }

  void setInference(bool inference) { inferenceOnly_ = inference; }
  bool isInference() { return inferenceOnly_; }

  void setOptimized(bool optimized) { optimized_ = optimized; }
  bool isOptimized() { return (optimized_ && inferenceOnly_); }

  void setCheckpointing(bool checkpointing) { checkpointing_ = checkpointing; }
  bool isCheckpointing() { return checkpointing_; }

  void switchParams(const std::string& newNamespace) {
    namespace_ = newNamespace;
  }

  virtual void copyParams(Ptr<ExpressionGraph> graph) {
    for(auto p : *graph->params())
      param(p->name(), p->shape(), inits::fromTensor(p->val()), p->value_type());
    forward(); // this will allocate parameters, execute the intializers and therefore copy parameter values
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

  void checkNaN(Tensor t, bool& isNaN, bool& isInf);

  void forward() {
    params_->allocateForward();
    forwardNext();
  }

  void forwardNext();
  void forward(std::list<Expr>& forwardTape, bool finalPass);

  void backward(bool reset = true, float clipValue = 0.f);

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
             const Ptr<inits::NodeInitializer>& init,
             const Type valueType,
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
    p = Expression<ParamNode>(shared_from_this(), shape, init, valueType, fixed);

    // set name and id and add to list of parameters
    p->set_name(name);
    params_->add(p, name);

    return p;
  }

  Expr param(const std::string& pname,
             const Shape& shape,
             const Ptr<inits::NodeInitializer>& init,
             bool fixed = false) {
    return param(pname, shape, init, parameterType_, fixed);
  }

  Expr constant(const Shape& shape,
                const Ptr<inits::NodeInitializer>& init,
                Type valueType) {
    return Expression<ConstantNode>(shared_from_this(), shape, init, valueType);
  }

  Expr constant(const Shape& shape,
                const Ptr<inits::NodeInitializer>& init) {
    return Expression<ConstantNode>(shared_from_this(), shape, init, parameterType_);
  }

  // @TODO: add version with iterators
  // shortcut to turn vector of indices to integer tensor, to be used with operators
  // like rows or select
  Expr indices(const std::vector<IndexType>& indicesVector) {
    return constant({(int)indicesVector.size()},
                    inits::fromVector(indicesVector),
                    Type::uint32);
  }
  // this version sets up the shape such that the indices are in a given axis
  // Use this if you want to pass these indices to gather().
  // indexee shape = (3, 2, 5, 2); axis = 1 -> resulting shape = (1, size of indicesVector, 1, 1)
  Expr indices(const std::vector<IndexType>& indicesVector, Expr indexee, int axis = -1) {
    Shape shape;
    shape.resize(indexee->shape().size());
    shape.set(axis, indicesVector.size());
    return constant(Shape(shape),
                    inits::fromVector(indicesVector),
                    Type::uint32);
  }

  Expr ones(const Shape& shape, Type valueType) {
    return constant(shape, inits::ones(), valueType);
  }
  Expr ones(const Shape& shape) {
    return constant(shape, inits::ones(), parameterType_);
  }

  Expr zeros(const Shape& shape, Type valueType) {
    return constant(shape, inits::zeros(), valueType);
  }
  Expr zeros(const Shape& shape) {
    return constant(shape, inits::zeros(), parameterType_);
  }

  // prob = dropProb, e.g. 0.1 means 90% of values are kept
  Expr dropoutMask(float dropProb, const Shape& shape, Type valueType);
  Expr dropoutMask(float dropProb, const Shape& shape);

  Expr get(std::string name) {
    if(!namespace_.empty())
      name = namespace_ + "::" + name;

    auto e = params_->get(name);
    if(e)
      return e;
    return Expr();  // @TODO: how is this different from just returning 'e'?
  }

  Ptr<Parameters>& params() { return params_; }

  Type getParameterType() {
    return parameterType_;
  }

  Type getSaveType() {
    return parameterType_;
  }

  void setParameterType(Type parameterType, Type saveType = Type::float32) {
    parameterType_ = parameterType;
    saveType_ = saveType;
  }

  Expr add(Expr node);

  void allocateForward(Expr node) {
    if(tensors_)
      tensors_->allocateForward(node);
  }

  void allocateBackward(Expr node) {
    if(tensors_)
      tensors_->allocateBackward(node);
  }

  void free(const Tensor& tensor) {
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
  bool getThrowNaN() { return throwNaN_; }

public:
  // loading from array of io::Items
  void load(std::vector<io::Item>& ioItems, bool markReloaded = true) {
    setReloaded(false);
    for(auto& item : ioItems) {
      std::string pName = item.name;
      // skip over special parameters starting with "special:"
      if(pName.substr(0, 8) == "special:")
        continue;
      param(pName, item.shape, inits::fromItem(item));
    }
    if(markReloaded)
      setReloaded(true);
  }

  void load(const std::string& name, bool markReloaded = true) {
    LOG(info, "Loading model from {}", name);
    auto items = io::loadItems(name);
    load(items, markReloaded);
  }

  void load(const void* ptr, bool markReloaded = true) {
    LOG(info, "Loading model from buffer at {}", ptr);
    auto items = io::loadItems(ptr);
    load(items, markReloaded);
  }

  void mmap(const void* ptr, bool markReloaded = true) {
    ABORT_IF(backend_->getDeviceId().type != DeviceType::cpu || !inferenceOnly_,
             "Memory mapping only supported for CPU inference mode");

    params_ = New<MappedParameters>();
    params_->init(backend_);

    LOG(info, "Memory mapping model at {}", ptr);
    auto items = io::mmapItems(ptr);
    load(items, markReloaded);
  }

public:
  // convert all parameters into an array of io::Item elements, for saving
  void save(std::vector<io::Item>& ioItems);

  void save(const std::string& name, const std::string& meta = "") {
    // LOG(info, "Saving model to {}", name);

    std::vector<io::Item> ioItems;
    save(ioItems);
    if(!meta.empty())
      io::addMetaToItems(meta, "special:model.yml", ioItems);
    io::saveItems(name, ioItems);

    // LOG(info, "Saved {} items.", ioItems.size());
  }
};

template <class T, typename... Args>
Expr Expression(Args&&... args) {
  auto e = Expr(new T(std::forward<Args>(args)...));
  return e->graph()->add(e);
}
}  // namespace marian
