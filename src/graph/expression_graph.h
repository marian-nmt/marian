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

typedef std::map<Type, Ptr<Parameters>> ElementTypeParamsMap; // keep it sorted, hence map not unordered map

class ExpressionGraph : public std::enable_shared_from_this<ExpressionGraph> {
  size_t count_{0};

  std::list<Expr> nodesForward_;
  std::list<Expr> nodesBackward_;

  std::unordered_set<Expr> topNodes_; // current set of roots. In the end, all but one must have been consumed.

  // Holds memory and expressions that correspond to temporary expressions.
  // This gets cleared before a new graph is built.
  Ptr<Tensors> tensors_;

  std::unordered_map<size_t, std::vector<Expr>> memoized_;

  Type defaultElementType_{Type::float32}; // Type used for storing parameters, currently all parameters have to have the same type

  bool inferenceOnly_{false};

  // during inference, use optimizations that might lead to precision loss, e.g. 8-bit MatMul.
  // At this moment, this is used for int16 qunatized Matmul - 11/1/2019
  bool optimized_{false};
  bool checkpointing_{false}; // use gradient checkpointing if true

  bool reloaded_{false};

  bool throwNaN_{false};

protected:
  // Delete, copy and move constructors
  ExpressionGraph(const ExpressionGraph&) = delete;
  ExpressionGraph(ExpressionGraph&&) = delete;

  // Holds memory and expressions that correspond to graph parameters
  // Now we can have multiple types of parameters in a separate parameters object per value type. 
  // This is currently only accessible through private functions during loading, will abort during training
  // when params() is called (e.g. optimizer) and there is more or other types than the default parameter type.
  // Currently the only usecase is inference. Trying to access params() for non-default parameter type is going
  // to abort. Inference does not need to access a whole set of parameters.
  ElementTypeParamsMap paramsByElementType_;
  Ptr<Backend> backend_;

  std::string namespace_;

public:
  /** @brief Constructs a new expression graph
   *
   * Constructor should be used as New<ExpressionGraph>()
   */
  ExpressionGraph(bool inference = false);

  virtual ~ExpressionGraph() {
    clear();
    for(auto kvParams : paramsByElementType_)
      kvParams.second->clear();
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
    for(auto kvParams : paramsByElementType_)
      kvParams.second->allocateForward();
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

private:

  // Find the named parameter and its typed parent parameter object (params) and return both.
  // If the parameter is not found return the parent parameter object that the parameter should be added to.
  // Return [nullptr, nullptr] if no matching parent parameter object exists. 
  std::tuple<Expr, Ptr<Parameters>> findParams(const std::string& name, 
                                               Type elementType, 
                                               bool typeSpecified) const {
    Expr p; Ptr<Parameters> params;
    if(typeSpecified) { // type has been specified, so we are only allowed to look for a parameter with that type
      auto it = paramsByElementType_.find(elementType);
      if(it != paramsByElementType_.end()) {
        params = it->second;
        p = params->get(name);
      }
    } else { // type has not been specified, so we take any type as long as the name matches
      for(auto kvParams : paramsByElementType_) {
        p = kvParams.second->get(name);
        
        if(p) { // p has been found, return with matching params object
          params = kvParams.second;
          break;
        }
        
        if(kvParams.first == elementType) // even if p has not been found, set the params object to be returned
          params = kvParams.second;
      }
    }

    return std::make_tuple(p, params);
  }

  Expr param(const std::string& pname,
             const Shape& shape,
             const Ptr<inits::NodeInitializer>& init,
             const Type elementType,
             bool fixed,
             bool typeSpecified) {
    std::string name = pname;
    if(!namespace_.empty())
      name = namespace_ + "::" + name;

    Expr p; Ptr<Parameters> params; std::tie
    (p, params) = findParams(name, elementType, typeSpecified);
    
    if(!params) { 
      params = New<Parameters>(elementType);
      params->init(backend_);
      paramsByElementType_.insert({elementType, params});
    } else {
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
    }

    // if graph was reloaded do not allow creation of new parameters
    ABORT_IF(reloaded_,
             "Graph was reloaded and parameter '{}' with type {} (specified: {}) is newly created",
             name, elementType, typeSpecified);

    // if not check if name is not taken by other node
    auto other = get(name);
    ABORT_IF(other, "Parameter with name '{}' already exists and has type {}", name, other->value_type());

    // create parameter node (adds to tape)
    p = Expression<ParamNode>(shared_from_this(), shape, init, elementType, fixed);
    LOG(debug, "Created parameter {} with shape {} and type {}", name, shape, elementType);

    // set name and id and add to list of parameters
    p->set_name(name);
    params->add(p, name);

    return p;
  }

public:
  Expr param(const std::string& pname,
             const Shape& shape,
             const Ptr<inits::NodeInitializer>& init,
             const Type elementType,
             bool fixed = false) {
    // since this param is called with out a specified type, we assume defaultElementType but allow to check for a different type
    return param(pname, shape, init, elementType, fixed, /*typeSpecified=*/true);
  }

  Expr param(const std::string& pname,
             const Shape& shape,
             const Ptr<inits::NodeInitializer>& init,
             bool fixed = false) {
    // since this param is called with out a specified type, we assume defaultElementType but allow to check for a different type
    return param(pname, shape, init, defaultElementType_, fixed, /*typeSpecified=*/false);
  }

  Expr constant(const Shape& shape,
                const Ptr<inits::NodeInitializer>& init,
                Type elementType) {
    return Expression<ConstantNode>(shared_from_this(), shape, init, elementType);
  }

  Expr constant(const Shape& shape,
                const Ptr<inits::NodeInitializer>& init) {
    return Expression<ConstantNode>(shared_from_this(), shape, init, defaultElementType_);
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

  Expr ones(const Shape& shape, Type elementType) {
    return constant(shape, inits::ones(), elementType);
  }
  Expr ones(const Shape& shape) {
    return constant(shape, inits::ones(), defaultElementType_);
  }

  Expr zeros(const Shape& shape, Type elementType) {
    return constant(shape, inits::zeros(), elementType);
  }
  Expr zeros(const Shape& shape) {
    return constant(shape, inits::zeros(), defaultElementType_);
  }

  // prob = dropProb, e.g. 0.1 means 90% of values are kept
  Expr dropoutMask(float dropProb, const Shape& shape, Type elementType);
  Expr dropoutMask(float dropProb, const Shape& shape);

  Expr get(std::string name) {
    if(!namespace_.empty())
      name = namespace_ + "::" + name;
    Expr p; Ptr<Parameters> params; std::tie
    (p, params) = findParams(name, defaultElementType_, /*specifiedType=*/false);
    return p;
  }

  Expr get(std::string name, Type specifiedElementType) {
    if(!namespace_.empty())
      name = namespace_ + "::" + name;
    Expr p; Ptr<Parameters> params; std::tie
    (p, params) = findParams(name, specifiedElementType, /*specifiedType=*/true);
    return p;
  }

  Ptr<Parameters>& params() { 
    // There are no parameter objects, that's weird.
    ABORT_IF(paramsByElementType_.empty(), "No parameter object has been created");
    
    // Safeguard against accessing parameters from the outside with multiple parameter types, not yet supported
    ABORT_IF(paramsByElementType_.size() > 1, "Calling of params() is currently not supported with multiple ({}) parameters", paramsByElementType_.size());
    
    // Safeguard against accessing parameters from the outside with other than default parameter type, not yet supported
    auto it = paramsByElementType_.find(defaultElementType_);
    ABORT_IF(it == paramsByElementType_.end(), "Parameter object for type {} does not exist", defaultElementType_);

    return it->second;
  }

  void setDefaultElementType(Type defaultElementType) {
    ABORT_IF(!paramsByElementType_.empty() && defaultElementType != defaultElementType_, 
             "Parameter objects already exist, cannot change default type from {} to {}", 
             defaultElementType_, defaultElementType);
    defaultElementType_ = defaultElementType;
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
      param(pName, item.shape, inits::fromItem(item), item.type, /*fixed=*/false);
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

    LOG(info, "Memory mapping model at {}", ptr);
    auto items = io::mmapItems(ptr);
    
    // Deal with default parameter set object that might not be a mapped object.
    // This gets assigned during ExpressionGraph::setDevice(...) and by default 
    // would contain allocated tensors. Here we replace it with a mmapped version.
    auto it = paramsByElementType_.find(defaultElementType_);
    if(it != paramsByElementType_.end()) {
      // there is parameter object for that type
      auto defaultParams = std::dynamic_pointer_cast<MappedParameters>(it->second);
      if(!defaultParams) {
        // but it's not mapped, so delete it and replace it with a mapped version
        defaultParams = New<MappedParameters>(defaultElementType_);
        defaultParams->init(backend_);
        paramsByElementType_[defaultElementType_] = defaultParams;
      }
    }


    // pre-populate parameters by type
    for(auto& item : items) {
      auto it1 = paramsByElementType_.find(item.type);
      if(it1 == paramsByElementType_.end()) {
        auto params = New<MappedParameters>(item.type);
        params->init(backend_);
        paramsByElementType_.insert({item.type, params});
      }
    }

    load(items, markReloaded);
  }

public:
  // convert all parameters into an array of io::Item elements, for saving
  void save(std::vector<io::Item>& ioItems, Type saveElementType = Type::float32);

  void save(const std::string& name, const std::string& meta = "", Type saveElementType = Type::float32) {
    std::vector<io::Item> ioItems;
    save(ioItems, saveElementType);
    if(!meta.empty())
      io::addMetaToItems(meta, "special:model.yml", ioItems);
    io::saveItems(name, ioItems);
  }
};

template <class T, typename... Args>
Expr Expression(Args&&... args) {
  auto e = Expr(new T(std::forward<Args>(args)...));
  return e->graph()->add(e);
}
}  // namespace marian
