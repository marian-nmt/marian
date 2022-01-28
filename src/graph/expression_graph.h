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

/**
 * Create an expression node of any type, and pass all
 * arguments to any available constructor.
 * E.g., to create a ConstantNode uses `Expression<ConstantNode>(...)`.
 */
template <class T, typename... Args>
Expr Expression(Args&&... args);

/**
 * The whole tensor set in the graph.
 * Holds all tensor objects (memory and nodes) for a graph.
 */
class Tensors {
private:
  Ptr<TensorAllocator> tensors_;
  Ptr<TensorAllocator> cache_;

  typedef std::unordered_map<size_t, std::vector<WExpr>> WeakMemory;
  typedef std::unordered_map<size_t, std::vector<Expr>> Memory;

  Ptr<WeakMemory> shortterm_;  // holds all nodes for a graph
  Ptr<Memory> longterm_;  // holds memoized nodes

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

  Ptr<Allocator>       getAllocator() { return tensors_->allocator(); }
  Ptr<TensorAllocator> getTensorAllocator() { return tensors_; }

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

/**
 *  Main implementation of a computation graph.
 *  Keeps a record of data (tensors) and all operations. Each operation in a computation graph is a Node.
 *  Each Node defines its forward and backward steps.
 */
class ExpressionGraph : public std::enable_shared_from_this<ExpressionGraph> {
  size_t count_{0};  // counter for nodes in the graph; hold current node index

  std::unordered_set<Expr> topNodes_; // current set of roots. In the end, all but one must have been consumed

protected:  // (these are protected, not private, for ONNX exporting)
  std::list<Expr> nodesForward_;     ///< contains all nodes used for forward()
  std::list<Expr> nodesBackward_;    ///< contains trainable nodes used for backward()

  /**
   * A shared pointer to the tensor objects in the graph.
   * Holds memory and nodes that corresponds to tensors in a graph.
   * Since operations will result in new tensors, this attribute is used
   * to allocate memory for new tensors during forward() and backward().
   * This gets cleared before a new graph is built.
   */
  Ptr<Tensors> tensors_;
private:

  Type defaultElementType_{Type::float32};  // Type used for storing parameters, currently all parameters have to have the same type

  bool inferenceOnly_{false};               // a flag holds whether the graph is used for inference only

  bool checkpointing_{false};               // use gradient checkpointing if true

  bool reloaded_{false};                    // a flag holds whether the graph is reloaded: reloaded is true if the graph loads parameters by load() function.

  bool throwNaN_{false};                    // a flag holds whether the graph throws a NaN exception

protected:
  // Delete, copy and move constructors
  ExpressionGraph(const ExpressionGraph&) = delete;
  ExpressionGraph(ExpressionGraph&&) = delete;

  /**
   * A map holds memory and nodes that corresponds to graph parameters.
   * The key is Type and the mapped value is a set of parameter objects with corresponding type.
   * Now we can have multiple types of parameters in a separate parameters object per value type.
   * This is currently only accessible through private functions during loading, will abort during training
   * when params() is called (e.g. optimizer) and there is more or other types than the default parameter type.
   * Currently the only usecase is inference. Trying to access params() for non-default parameter type is going
   * to abort. Inference does not need to access a whole set of parameters.
   */
  ElementTypeParamsMap paramsByElementType_;
  Ptr<Backend> backend_;      ///< a shared pointer to the backend for the graph
  std::string namespace_;     ///< a string defines the namespace of the graph. Each graph has its own unique namespace.

public:
  /** Constructs a new expression graph. Constructor should be used as New<ExpressionGraph>(). */
  ExpressionGraph(bool inference = false);

  /** Destructor. Clear everything related to the graph except memoized nodes. */
  virtual ~ExpressionGraph() {
    clear();
    for(auto kvParams : paramsByElementType_)
      kvParams.second->clear();
  }

  /**
   * Set device options used to run the graph.
   * @param deviceId a struct type which stores device no. (size_t)
   * and device type (DeviceType::cpu or DeviceType::gpu)
   * @param device a pointer to the device
   */
  virtual void setDevice(DeviceId deviceId = {0, DeviceType::gpu},
                         Ptr<Device> device = nullptr);

  /**
   * Get device info for the graph.
   * @return deviceId a struct type which stores device no. (size_t)
   * and device type (DeviceType::cpu or DeviceType::gpu)
   */
  DeviceId getDeviceId() { return backend_->getDeviceId(); }

  /**
   * Get backend pointer for the graph.
   * @return Ptr<Backend> pointer to backend
   */
  Ptr<Backend> getBackend() { return backend_; }

  /** Set whether the graph is used for inference only */
  void setInference(bool inference) { inferenceOnly_ = inference; }

  /** Check whether the graph is used for inference only (true) or not */
  bool isInference() { return inferenceOnly_; }

  /**
   * Set whether the graph uses gradient checkpointing.
   * <a href="https://github.com/cybertronai/gradient-checkpointing">Gradient Checkpointing</a>
   * works by trading compute for memory, which reruns a forward-pass segment for each checkpoint segment during backward.
   */
  void setCheckpointing(bool checkpointing) { checkpointing_ = checkpointing; }

  /** Check whether the graph uses gradient checkpointing or not */
  bool isCheckpointing() { return checkpointing_; }

  /**
   * Set namespace (std::string) for the graph.
   * Each graph has its own unique namespace, which is used to form the name of a parameter object.
   */
  void switchParams(const std::string& newNamespace) {
    namespace_ = newNamespace;
  }

  /**
   * Copy all parameter objects from one graph to current graph.
   * @param graph a pointer to a graph object
   */
  virtual void copyParams(Ptr<ExpressionGraph> graph) {
    for(auto p : *graph->params())
      param(p->name(), p->shape(), inits::fromTensor(p->val()), p->value_type());
    forward(); // this will allocate parameters, execute the initializers and therefore copy parameter values
  }

  /**
   * Preallocate workspace memory (MB) for the graph.
   * Sets the size of the memory available for the forward and backward step of the training procedure.
   * This does not include model size and optimizer parameters that are allocated outsize workspace.
   */
  void reserveWorkspaceMB(size_t num) {
    size_t bytes = num * 1024 * 1024 - 1;
    tensors_->reserve(bytes);
  }

  /** Copy tensor objects from one graph to current graph */
  void reuseWorkspace(Ptr<ExpressionGraph> graph) {
    tensors_ = graph->tensors_;
  }

  /**
   * Performs backpropagation on this expression graph.
   * Backpropagation is implemented by performing first the forward pass and
   * then the backward pass of algorithmic differentiation (AD) on the nodes of
   * the graph.
   */
  void backprop() {
    forward();
    backward();
  }

  /**
   * Perform one backpropagation process on the graph to test
   * whether the graph workspace fits into a given workspace memory.
   * This function is used for searching the maximum batch size
   * that fits into given workspace memory.
   */
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

  /**
   * Check whether the memory allocated for a tensor object contains a NaN or infinite value.
   * @param t a Tensor object
   * @param isNaN a bool type holds the result whether the tensor contains a NaN value (pass by reference)
   * @param isInf a bool type holds the result whether the tensor contains a infinite value (pass by reference)
   */
  void checkNaN(Tensor t, bool& isNaN, bool& isInf);

  /**
   * Perform the forward pass on the nodes of the graph.
   * The forward pass refers to the calculation process.
   * It traverses through all nodes from input layer to output layer.
   */
  void forward() {
    for(auto kvParams : paramsByElementType_)
      kvParams.second->allocateForward();
    forwardNext();
  }

  /**
   * Perform the forward pass without memory allocation for parameters.
   * Helper function for forward().
   */
  void forwardNext();

  /**
   * Perform forward pass on a given nodes with finalPass flag.
   * Helper function for forward() and backward().
   * @param forwardTape a pointer to the nodes used for forward pass
   * @param finalPass a bool type which controls whether nodes should be freed with gradient-checkpointing
   */
  void forward(std::list<Expr>& forwardTape, bool finalPass);

  /**
   * Perform the backward pass on the trainable nodes of the graph.
   * The back pass refers to the process of computing the output error.
   * It traverses through all nodes from output layer to input layer.
   */
  void backward(bool reset = true, float clipValue = 0.f);

  /**
   * Generate graph layout in Graphviz format for visualisation.
   * @return a string presenting graph layout in Graphviz format (dot)
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

  /**
   * Write graph layout in Graphviz format to a file.
   * @param filename a string type specifies filename that writes the graph layout
   */
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

  /**
   * Construct a parameter node in the graph.
   * @param pname a string type holds the name of the parameter node
   * @param shape a struct type defines the shape of the parameter tensor
   *        e.g., shape={2,3} means 2D matrix with dim[0]=2 and dim[1]=3
   * @param init a pointer to a NodeInitializer object, e.g., inits::zeros()
   * @param elementType a scoped enumerator (enum class) defines the element type, e.g., Type::float16
   * @param fixed a bool type specifies whether the parameter object is fixed (not trainable) or not.
   *        The default value is false which means the parameter is trainable.
   * @return a pointer to the parameter node
   */
  Expr param(const std::string& pname,
             const Shape& shape,
             const Ptr<inits::NodeInitializer>& init,
             const Type elementType,
             bool fixed = false) {
    // this param is called with a specified type
    return param(pname, shape, init, elementType, fixed, /*typeSpecified=*/true);
  }

  /**
   * Construct a parameter node in the graph without a specified type, and
   * the type is set to defaultElementType_.
   * @param pname a string type holds the name of the parameter node
   * @param shape a struct type defines the shape of the parameter tensor
   *        e.g., shape={2,3} means 2D matrix with dim[0]=2 and dim[1]=3
   * @param init a pointer to a NodeInitializer object, e.g., inits::zeros()
   * @param fixed a bool type specifies whether the parameter object is fixed (not trainable) or not.
   *        The default value is false which means the parameter is trainable.
   * @return a pointer to the parameter node
   */
  Expr param(const std::string& pname,
             const Shape& shape,
             const Ptr<inits::NodeInitializer>& init,
             bool fixed = false) {
    // since this param is called without a specified type, we assume defaultElementType but allow to check for a different type
    return param(pname, shape, init, defaultElementType_, fixed, /*typeSpecified=*/false);
  }

  /**
   * Construct a constant node in the graph without a specified type, and
   * the type is set to defaultElementType_.
   * @param shape a struct type defines the shape of the constant tensor
   *        e.g., shape={2,3} means 2D matrix with dim[0]=2 and dim[1]=3
   * @param init a pointer to a NodeInitializer object, e.g., inits::zeros()
   * @param elementType a scoped enumerator (enum class) defines the element type, e.g., Type::float16
   * @return a pointer to the constant node
   */
  Expr constant(const Shape& shape,
                const Ptr<inits::NodeInitializer>& init,
                Type elementType) {
    return Expression<ConstantNode>(shared_from_this(), shape, init, elementType);
  }

  /**
   * Construct a constant node in the graph without a specified type, and
   * the type is set to defaultElementType_.
   * @param shape a struct type defines the shape of the constant tensor
   *        e.g., shape={2,3} means 2D matrix with dim[0]=2 and dim[1]=3
   * @param init a pointer to a NodeInitializer object, e.g., inits::zeros()
   * @return a pointer to the constant node
   */
  Expr constant(const Shape& shape,
                const Ptr<inits::NodeInitializer>& init) {
    return Expression<ConstantNode>(shared_from_this(), shape, init, defaultElementType_);
  }

  // @TODO: add version with iterators
  /**
   * Turn vector of indices to integer tensor.
   * A shortcut version to turn vector of indices to integer tensor, to be used with operators
   * like rows() or index_select()
   * @param indicesVector a vector of IndexType (uint32_t) specifies the indexes
   */
  Expr indices(const std::vector<IndexType>& indicesVector) {
    return constant({(int)indicesVector.size()},
                    inits::fromVector(indicesVector),
                    Type::uint32);
  }

  /**
   * Specify the indexes of elements to be taken from a tensor.
   * This version sets up the shape such that the indices are in a given axis.
   * Use this if you want to pass these indices to gather().
   * E.g., indexee shape = (3, 2, 5, 2); axis = 1 -> resulting shape = (1, size of indicesVector, 1, 1):
   *  - The size of the resulting shape is the same as that of the indexee; here is 4.
   *  - The shape of the specified axis is equal to the size of given indicesVector.
   *  - The shapes of the rest axes are filled with 1.
   * @param indicesVector a vector of IndexType (uint32_t) specifies the indexes
   * @param indexee the source tensor that we want to select elements from
   * @param axis specifies the axis that we want to collect along
   */
  Expr indices(const std::vector<IndexType>& indicesVector, Expr indexee, int axis = -1) {
    Shape shape;
    shape.resize(indexee->shape().size());
    shape.set(axis, indicesVector.size());
    return constant(Shape(shape),
                    inits::fromVector(indicesVector),
                    Type::uint32);
  }

  /**
   * Construct a constant node filled with `1`.
   * @param shape a struct type defines the shape of the constant dataset
   *        e.g., shape={2,3} means 2D matrix with dim[0]=2 and dim[1]=3
   * @param elementType a scoped enumerator (enum class) defines the element type, e.g., Type::float16
   */
  Expr ones(const Shape& shape, Type elementType) {
    return constant(shape, inits::ones(), elementType);
  }

  /**
   * Construct a constant node filled with `1` without a specified type,
   * and the type is set to defaultElementType_.
   * @param shape a struct type defines the shape of the constant dataset
   *        e.g., shape={2,3} means 2D matrix with dim[0]=2 and dim[1]=3
   */
  Expr ones(const Shape& shape) {
    return constant(shape, inits::ones(), defaultElementType_);
  }

  /**
   * Construct a constant node filled with `0`.
   * @param shape a struct type defines the shape of the constant dataset
   *        e.g., shape={2,3} means 2D matrix with dim[0]=2 and dim[1]=3
   * @param elementType a scoped enumerator (enum class) defines the element type, e.g., Type::float16
   */
  Expr zeros(const Shape& shape, Type elementType) {
    return constant(shape, inits::zeros(), elementType);
  }

  /**
   * Construct a constant node filled with `0` without a specified type,
   * and the type is set to defaultElementType_.
   * @param shape a struct type defines the shape of the constant dataset
   *        e.g., shape={2,3} means 2D matrix with dim[0]=2 and dim[1]=3
   */
  Expr zeros(const Shape& shape) {
    return constant(shape, inits::zeros(), defaultElementType_);
  }

  /**
   * Construct a dropout mask (a tensor of 0 and 1).
   * @param dropProb a float type specifies the dropout probability.
   *        E.g., dropProb=0.1 means 90% of values are kept.
   * @param shape a struct type defines the shape of the constant dataset
   *        e.g., shape={2,3} means 2D matrix with dim[0]=2 and dim[1]=3
   * @param elementType a scoped enumerator (enum class) defines the element type, e.g., Type::float16
   */
  Expr dropoutMask(float dropProb, const Shape& shape, Type elementType);

  /**
   * Construct a dropout mask (a tensor of 0 and 1) without a specified type,
   * and the type is set to defaultElementType_.
   * @param dropProb a float type specifies the dropout probability.
   *        E.g., dropProb=0.1 means 90% of values are kept.
   * @param shape a struct type defines the shape of the constant dataset
   *        e.g., shape={2,3} means 2D matrix with dim[0]=2 and dim[1]=3
   */
  Expr dropoutMask(float dropProb, const Shape& shape);

  /**
   * Get the parameter object by name.
   * @param name a string specifies the name of the parameter object
   */
  Expr get(std::string name) {
    if(!namespace_.empty())
      name = namespace_ + "::" + name;
    Expr p; Ptr<Parameters> params; std::tie
    (p, params) = findParams(name, defaultElementType_, /*specifiedType=*/false);
    return p;
  }

  /**
   * Get the parameter object by name and type.
   * @param name a string specifies the name of the parameter object
   * @param elementType a scoped enumerator (enum class) defines the element type, e.g., Type::float16
   */
  Expr get(std::string name, Type specifiedElementType) {
    if(!namespace_.empty())
      name = namespace_ + "::" + name;
    Expr p; Ptr<Parameters> params; std::tie
    (p, params) = findParams(name, specifiedElementType, /*specifiedType=*/true);
    return p;
  }

  /**
   * Return the Parameters object related to the graph.
   * The Parameters object holds the whole set of the parameter nodes.
   */
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

  /**
   * Return the Parameters object related to the graph by elementType.
   * The Parameters object holds the whole set of the parameter nodes of the given type.
   */
  Ptr<Parameters>& params(Type elementType) {
    auto it = paramsByElementType_.find(elementType);
    ABORT_IF(it == paramsByElementType_.end(), "Parameter object for type {} does not exist", defaultElementType_);
    return it->second;
  }

  /**
   * Set default element type for the graph.
   * The default value is used if some node type is not specified.
   */
  void setDefaultElementType(Type defaultElementType) {
    ABORT_IF(!paramsByElementType_.empty() && defaultElementType != defaultElementType_,
             "Parameter objects already exist, cannot change default type from {} to {}",
             defaultElementType_, defaultElementType);
    defaultElementType_ = defaultElementType;
  }

  /**
   * Get default element type for the graph.
   */
  Type getDefaultElementType() { return defaultElementType_; }

  /**
   * Add a expression node to the graph.
   * @param node a pointer to a expression node
   */
  Expr add(Expr node);

  /**
   * Allocate memory for the forward pass of the given node.
   * @param node a pointer to a expression node
   */
  void allocateForward(Expr node) {
    if(tensors_)
      tensors_->allocateForward(node);
  }

  /**
   * Allocate memory for the backward pass of the given node.
   * @param node a pointer to a expression node
   */
  void allocateBackward(Expr node) {
    if(tensors_)
      tensors_->allocateBackward(node);
  }

  /**
   * Free the memory for a tensor object.
   * @param tensor a reference to the tensor object
   */
  void free(const Tensor& tensor) {
    if(tensors_)
      tensors_->free(tensor);
  }

  /**
   * Returns the memory allocator of the graph workspace.
   * Allocates raw unstructured memory (but 256-byte aligned).
   */
  Ptr<Allocator> allocator() { return tensors_->getAllocator(); } // @TODO: rename this to getAllocator();

  /**
   * Returns the tensor allocator of the graph workspace.
   * Different from allocator() as proper tensor objects are allocated.
   */
  Ptr<TensorAllocator> getTensorAllocator() { return tensors_->getTensorAllocator(); }

  /** Clear everything apart from parameters and memoized nodes */
  void clear() {
    count_ = 0;
    nodesForward_.clear();
    nodesBackward_.clear();

    topNodes_.clear();

    tensors_->clear();
  }

  /** Set the flag value whether the graph is reloaded (true) or not */
  void setReloaded(bool reloaded) { reloaded_ = reloaded; }

  /** Set the flag value whether the graph throws a NaN exception (true) or not */
  void setThrowNaN(bool throwNaN) { throwNaN_ = throwNaN; }

  /** Get the flag value whether the graph throws a NaN exception (true) or not */
  bool getThrowNaN() { return throwNaN_; }

public:
  /** Load model (mainly parameter objects) from array of io::Items */
  void load(const std::vector<io::Item>& ioItems, bool markReloaded = true) {
    setReloaded(false);
    for(auto& item : ioItems) {
      std::string pName = item.name;
      // skip over special parameters starting with "special:"
      if(pName.substr(0, 8) == "special:")
        continue;

      // if during loading the loaded type is of the same type class as the default element type, allow conversion;
      // otherwise keep the loaded type. This is used when e.g. loading a float32 model as a float16 model as both
      // have type class TypeClass::float_type.
      auto loadElementType = isSameTypeClass(item.type, defaultElementType_) ? defaultElementType_ : item.type;
      param(pName, item.shape, inits::fromItem(item), loadElementType, /*fixed=*/false);
    }
    if(markReloaded)
      setReloaded(true);
  }

  /** Load model by filename */
  void load(const std::string& name, bool markReloaded = true) {
    LOG(info, "Loading model from {}", name);
    auto items = io::loadItems(name);
    load(items, markReloaded);
  }

  /** Load model from buffer (a file pointer) */
  void load(const void* ptr, bool markReloaded = true) {
    LOG(info, "Loading model from buffer at {}", ptr);
    auto items = io::loadItems(ptr);
    load(items, markReloaded);
  }

  /**
   * Turn the model (given a file pointer) into a memory-mapped type
   * by converting all the parameter object to memory-mapped version, i.e., MappedParameters.
   */
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
  /**
   * Convert all parameters into an array of io::Item elements, for saving.
   * @param ioItems an array of io::Item elements
   * @param saveElementType the element type for saving
   */
  void save(std::vector<io::Item>& ioItems, Type saveElementType = Type::float32);

  /**
   * Save all parameters into a file (.npz or .bin).
   * @param name a string specifies the filename
   * @param meta a string specifies the name of io::Item elements. If not specified, the parameter name is reserved.
   * @param saveElementType the element type for saving
   */
  void save(const std::string& name, const std::string& meta = "", Type saveElementType = Type::float32) {
    std::vector<io::Item> ioItems;
    save(ioItems, saveElementType);
    if(ioItems.empty()) {
      LOG(warn, "Item list is empty, skipping saving");
    } else {
      if(!meta.empty())
        io::addMetaToItems(meta, "special:model.yml", ioItems);
      io::saveItems(name, ioItems);
    }
  }
};

template <class T, typename... Args>
Expr Expression(Args&&... args) {
  auto e = Expr(new T(std::forward<Args>(args)...));
  return e->graph()->add(e);
}
}  // namespace marian
