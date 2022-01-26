# Operations in the expression graph

Operations are responsible for manipulating the elements of an expression graph.
In Marian, many useful operations have already been implemented and can be found
the code documentation. The provided operations cover simple arithmetic, logical
comparisons and common mathematical functions; as well as tensor manipulation,
for example `slice` or `reshape`, and aggregations such as `sum` or `minimum`.
Finally, other routines, such as activation functions, useful in building
neutral networks are also available.

There are several necessary components required to implement an operation in
Marian's expression graph. The highest-level component is the Expression
Operator, responsible for setting up the Node Operator and adding it to the
graph. Next, this Node Operator describes the nature of the forward and backward
operation to be performed. These operations are implemented using some
combination of Functional Operators (element wise), and Tensor Operators.

This overview aims to provide information about what each of the different
operator components does, how they fit together and where to go to make changes.
Then, equipped with this knowledge, to be able to add new functionality to
Marian.

## Operator Structure

The central component in the graph is the `Chainable<Tensor>` object. This
object provides the abstract interface necessary to interact with elements in
the computation graph. The details of this interface can be found in
[/src/graph/chainable.h](file_src_graph_chainable.h). Note that the
template parameter corresponds to the underlying data structure, which in Marian
is the `Tensor`. Therefore, for convenience, the type `Expr` is defined:

```cpp
typedef IPtr<Chainable<Tensor>> Expr;
```

The implementation of the different operator components are divided across
several files:

  - Expression Operator
    - [/src/graph/expression_operators.h](file_src_graph_expression_operators.h)
    - [/src/graph/expression_operators.cpp](file_src_graph_expression_operators.cpp)
  - Node Operator
    - [/src/graph/node_operators_unary.h](file_src_graph_node_operators_unary.h)
    - [/src/graph/node_operators_binary.h](file_src_graph_node_operators_binary.h)
    - [/src/graph/node_operators_tuple.h](file_src_graph_node_operators_tuple.h)
  - Functional Operator
    - [/src/functional/operators.h](file_src_functional_operators.h)
  - Tensor operation
    - [/src/tensors/tensor_operators.h](file_src_tensors_tensor_operators.h)
    - [/src/tensors/cpu/tensor_operators.cpp](file_src_tensors_cpu_tensor_operators.cpp)
    - [/src/tensors/gpu/tensor_operators.cu](file_src_tensors_gpu_tensor_operators.cu)
  - Declared Specialization
    - [/src/tensors/gpu/element.inc](program_listing_file_src_tensors_gpu_element.inc)
    - [/src/tensors/gpu/add.inc](program_listing_file_src_tensors_gpu_add.inc)
    - [/src/tensors/gpu/add_all.inc](program_listing_file_src_tensors_gpu_add_all.inc)

To understand how the different components are inter-linked, we'll look at each
of them in turn.


## Expression Operator

The expression operator is the user-facing method used when building a graph. It
is responsible for constructing the corresponding Node Operation and inserting
it into the expression graph. To accommodate these core requirements, the
function `Expression` is able to perform both actions in generality:

```cpp
template <class T, typename... Args>
Expr Expression(Args&&... args) {
  auto e = Expr(new T(std::forward<Args>(args)...));
  return e->graph()->add(e);
}
```

This helper-function simplifies the definition of many expression operators. For
example, the implementation of the expression operator `sin(x)` is simply:

```cpp
// src/graph/expression_operators.h
Expr sin(Expr x);

// src/graph/expression_operators.cpp
Expr sin(Expr x) {
  return Expression<SinNodeOp>(x);
}
```

However, implementations may perform actions beyond the core functionality
alone. Taking `sum` as an example

```cpp
Expr sum(Expr a, int ax) {
  if(a->shape()[ax] == 1) {
    return a;
  }
  return Expression<ReduceNodeOp>(a, ax, ReduceNodeOpCode::sum);
}
```

The trivial operation is handled without needing to construct a node operation.
This example also demonstrates a non-trivial construction of `ReduceNodeOp`,
which is capable of performing differing reduction operations depending on
instantiation.

Going further, an expression operator may be defined in terms of existing
expressions. Operators such as `weighted_average` are composed of three
different expression operator calls: `scalar_product`, `sum`, and `operator/`.

```cpp
Expr weighted_average(Expr in, Expr weights, int ax) {
  auto p = scalar_product(in, weights, ax);
  auto s = sum(weights, ax);
  return p / s;
}
```

While useful, composition at this level may be less efficient than lower-level
implementations.


## Node Operator

The `Node` subclass of `Chainable<Tensor>` provides concrete implementations for
much of the abstract interface, while subclasses of `Node` enable different node
behaviours. In the context of operations, the relevant derived class is
`NaryNodeOp` and is base class used for Node Operators. This subclass provides
implementation focused on performing general N-arity operations. However, many
common operations are unary and, for convenience, a further specialization,
`UnaryNodeOp`, exists to simplify their definition.

The purpose of the Node Operator is to define the forward and backward behaviour
of the operation. The forward operation performs the desired operation while the
backward operation updates the gradients. These behaviours are written in terms
of `NodeOps`, where a `NodeOp` is a wrapper to define a capturing lambda
function. Explicitly these are defined as:

```cpp
// src/graph/chainable.h
#define NodeOp(op) [=]() { op; }
typedef std::vector<std::function<void()>> NodeOps;
```

Each `NodeOp` is written as a function in terms of the value (`val_`), gradient
(`adj_`) of the current node, and its children, via `child()`. The values and
gradients the n<sup>th</sup> child node are accessed via the interfaces
`child(n)->val()` and `child(n)->grad()`, respectively. NodeOps are executed in
order when running the graph forwards and backwards, as this snippet from `Node`
demonstrates

```cpp
// Node in src/graph/node.h
virtual void runForward(const NodeOps& ops) {
  for(auto&& op : ops)
    op();
}

virtual void runBackward(const NodeOps& ops) {
  size_t i = 0;
  for(auto&& op : ops)
    if(child(i++)->trainable())
      op();
}
```

In backwards operation it is **crucial** that the `NopeOp` responsible for
propagating a gradient to `child(i)` is the i<sup>th</sup> element of the
NodeOps vector. The requirement that the child associated with the NodeOp be
trainable means that an out-of-position NodeOp may not be run. To represent no
operation a `nullptr` can be passed as a NodeOp.

A typical node operator has the functionality demonstrated in the following
snippet.

```cpp
// outline of a node op
struct MyNodeOp : public NaryNodeOp {
  MyNodeOp(Expr a)
    : NaryNodeOp({a}, newShape(...), newType(...)) {}

  Shape newShape(...) {}  // optional
  Type newType(...) {}    // optional

  const std::string type() override { return "my_node_op"; }
  virtual size_t hash() override {}          // potentially required
  virtual bool equal(Expr node) override {}  // potentially required

  NodeOps forwardOps() override {}
  NodeOps backwardOps() override {}
```

This outline describes a node operator that takes a single argument `a`. The
shape and type of the node would be determined by the result of `newShape` and
`newType` when constructing the `NaryNodeOp`. These functions represent any
custom logic used to determine the shape and type of the node. As indicated in
this example code, these are optional and, when omitted, calling
`NaryNodeOp({a})` would result in a node with the same shape and type as `a`.
The `type()` method returns the friendly name for the node. Note that the
[ONNX](https://onnx.ai)
[interface](program_listing_file_src_onnx_expression_graph_onnx_serialization.cpp)
maintains a mapping of these friendly names to their ONNX representation. In the
absence of any member variables the `hash()` and `equal()` methods can be
omitted, and defer to their `NaryNodeOp` definition. However, if such variables
exist then `hash()` should implement a hashed representation and `equal()`
should provide the necessary conditions to consider nodes equivalent. Finally,
the operations of the node are defined in `forwardOps()` and `backwardOps()`.

Continuing with the example of `sin(x)`, the code responsible for implementing
the behaviour is

```cpp
// src/graph/node_operators_unary.h
struct SinNodeOp : public UnaryNodeOp {
  SinNodeOp(Expr x) : UnaryNodeOp(x) {}

  NodeOps forwardOps() override {
    using namespace functional;
    return {NodeOp(Element(_1 = sin(_2), val_, child(0)->val()))};
  }

  NodeOps backwardOps() override {
    using namespace functional;
    return {NodeOp(Add(_1 * cos(_2), child(0)->grad(), adj_, child(0)->val()))};
  }

  const std::string type() override { return "sin"; }
};
```

In this code, the constructor trivially initialises the `UnaryNodeOp`, passing
the expression `x` as its input. This propagates up to `NaryNodeOp` and becomes
`child(0)` of the node. The size and type of the SinNodeOp are equivalent to
that of `x`. The lack of any member variables allows the `hash()` and `equal()`
methods to be omitted. The friendly name for this node is the string `sin`. The
forward and backward implementation are accomplished using a single NodeOp each.

### Forward operation

The forward NodeOp calls the tensor operation Element, that execute the
element-wise operation described by the functor:

```cpp
_1 = sin(_2)
```

The placeholders `_1`, `_2` are enabled by code in
[/src/functional](dir_src_functional) and interoperate with the
functional operators. In the call to `Element`, `val_` is assigned to `_1` and
`child(0)->val()` to `_2`. Therefore, this has the action of setting the
elements of this node to the result obtained by applying `sin` to the elements
of `child(0)`.

### Backward Operation

The backward NodeOp is responsible for backpropagation of the gradients via
reverse-mode automatic differentiation. In this example, where `y = sin(x)`,
this corresponds to evaluating

```
dJ/dx += dJ/dy * dy/dx, dy/dx = cos(x)
```

This is realised using the tensor operator `Add` with the functor

```cpp
_1 * cos(_2)
```

In the call to `Add`, `adj_` is assigned to `_1` and `child(0)->val()` to `_2`.
Therefore, this functor represents `dJ/dy * dy/dx`: the product of the gradient
at the current node and the gradient of the operation. This value is then added
to the gradient of the child `child(0)->grad()` as required.

### Shape and Type Changes

The `newShape` and `newType` methods are just a suggestion of how custom logic
may be encapsulated where needed. However, in practice, many operations do not
require a change in shape or type. In these instances, the node inherits the
broadcasted shape of its children as well as their common type. An important
feature of the type deduction in `NaryNodeOp::commonType()` is that it
guarantees that all child nodes are of the same type.

There are few operations in Marian that require a type specification. Where they
do exist, they are often simple as the desired type is explicitly provided, or
is trivially deduced. An example of this is `CastNodeOp`

```cpp
// CastNodeOp in src/graph/node_operators_unary.h
CastNodeOp(Expr a, Type type) : UnaryNodeOp(a, type) {}
```

The desired type is set explicitly in construction. A slightly different example
is that of `CSRDotNodeOp`. It has several child nodes which are a mixture of
`DataType` and `IndexType` and therefore do not share a common type. The
solution is to explicitly specify the relevant children to
`NaryNodeOp::commonType({...})`.

Shape modifying operations are more common. A simple example is the class of
operations performed by `ReduceNodeOp` which involve an aggregation process
along one axis of the Tensor. The output shape is determined by

```cpp
// ReduceNodeOp in src/graph/node_operators_unary.h
Shape newShape(Expr a, int axis) {
  Shape shape = a->shape();
  axis_ = shape.axis(axis);

  shape.set(axis_, 1);
  return shape;
}
```

The output shape is the same as the input but with the processed axis is reduced
to a single element. Other use cases include transpose and slicing operations,
as well as tensor products.


## Functional Operator

As the NodeOp are evaluated, they encounter the underlying datatype of the
`Tensor`. At this stage, type-specific intrinsic functions are required. These
intrinsics are implemented in the templated struct `Ops<ElementType>`, with a
specialization required for each type. The current required types are:
  - float
  - double
  - float32x4 (see `src/3rd_party/sse_mathfun.h`)
  - float32x8 (see `src/3rd_party/avx_mathfun.h`)
  - half (see `cuda_fp16.h` in the CUDA Math API)

Further details are available in
[/src/common/types.h](file_src_common_types.h).

Returning to the example of `sin(x)`, the specialization for `float` and
`double` requires

```cpp
// src/functional/operators.h
// in namespace marian::functional
template <typename T>
struct Ops {
  static HOST_DEVICE_INLINE T sin(const T&)  { ABORT("Unknown type"); }
};

// Specialization for float
template <>
struct Ops<float> {
  static HOST_DEVICE_INLINE float sin(const float& x)  { return sinf(x); }
};

// Specialization for double
template <>
struct Ops<double> {
  static HOST_DEVICE_INLINE double sin(const double& x)  { return std::sin(x); }
};
```

The remaining specializations can be seen in
[/src/functional/operators.h](file_src_functional_operators.h). Note
that the general template must produce a runtime abort.

The final component of the functional operator is to call the macro that enables
interoperability with the framework of
[/src/functional](dir_src_functional). For a unary operator, this is
the macro `UNARY`.

```cpp
UNARY(Sin,     sin,        Ops<ElementType>::sin(x));
```

where template parameter `ElementType` **must** be used. There are equivalent
macros for `BINARY` and `TERNARY` Ops.


## Tensor Operator

Tensor operations use less abstracted interfaces to interact with the Tensors,
often working with the Tensor data directly. They also rely on BLAS (Basic
Linear Algebra Subprograms) libraries to accelerate these operations. As well as
libraries containing device-specific optimisations. These libraries include:

  - CPU
    - CBLAS / OpenBLAS
    - FBGEMM
    - INTGEMM
    - MKL
  - GPU
    - CUDA (cuBLAS)

An important subtlety is that while the CPU focused libraries use a row-major
representation, the cuBLAS library (GPU) instead uses a column-major
representation.

Furthermore, the OpenMPI and OpenMP libraries are employed for parallelisation.
While macros provided in
[/src/common/definitions.h](file_src_common_definitions.h) locally
enable faster floating-point math in supported compilers.

```cpp
MARIAN_FFAST_MATH_BEGIN
// ffmath code
MARIAN_FFAST_MATH_END
```

The usual caveats apply when enabling `fast_math`, and can be found in
[/src/common/definitions.h](file_src_common_definitions.h)

Tensor operators are declared in
[/src/tensors/tensor_operators.h](file_src_tensors_tensor_operators.h),
these are device-agnostic function that call the relevant device-specific
implementation. The CPU- and GPU-specific implementation are defined in `cpu`
namespace in [/src/tensors/cpu/](dir_src_tensors_cpu) and the `gpu`
namespace [/src/tensors/gpu/](dir_src_tensors_gpu). Therefore a typical
operator defers to an implementation in the device-specific namespace.

```cpp
void TensorOp(marian::Tensor out, marian::Tensor in) {
#ifdef CUDA_FOUND
  if(out->getBackend()->getDeviceId().type == DeviceType::gpu)
    gpu::TensorOp(out, in);
  else
#endif
    cpu::TensorOp(out, in);
}
```

When compiled with GPU support, this function dispatches a call to the
implementation that corresponds to the backend device type configured in the
graph (either GPU or CPU). Without GPU support, only the CPU implementation is
available.

Many operations are covered by three general tensor operators: `Element`,
`Aggregate` and `Prod`. The `Element` operator applies a function element-wise
across an arbitrary number of input tensors and stores the result in the output
tensor. The `Aggregate` operator also applies a function element-wise across its
inputs, but instead aggregates the results in the output via a given aggregation
function. A common aggregation function used is addition, which is the basis of
the `Add` and `Reduce` operators. Finally, `Prod` deals with products of
tensors. This operator performs a general matrix multiplication with the
underlying implementation relying on the libraries mentioned above.

Specialized operators exist to manipulation tensors beyond the cases covered
above; such as under transposition and concatenation. These operators may even
be expressed in terms of existing tensor operators.

Furthermore, for complicated multi-operation computations, performance gains and
memory improvements may be realised by implementing a tensor operator for that
specific purpose. An example of this is `softmax`, which could be implemented
using multiple expression operators (`exp`, `sum`), but is instead implemented
directly as a tensor operator. These optimized implementations may be device
specific.

## Declared Specialization

The operations performed in the forward and backward methods of NodeOp require
their GPU templates to be explicitly declared. When a new specialization is
introduced without being explicitly instantiated it will cause a link error on
compilation:

```
.../src/tensors/tensor_operators.h:41: undefined reference to `void marian::gpu::Element<marian::functional::Assign< ... > ( ... )'
```

To fix these undefined references, we must explicitly add the specialization to
the `.inc` files of [/src/tensors/gpu/](dir_src_tensors_gpu). Each
`.inc` file is included at the end of its corresponding `.cu` file, ensuring
that the specialization is compiled.

The undefined references should be added to the `.inc` file that corresponds to
the header file in which contains the declaration of the missing functions.

The file [element.inc](file_src_tensors_gpu_element.inc) contains the
specializations of the function defined in
[element.h](file_src_tensors_gpu_element.h):

```cpp
// src/tensors/gpu/element.h
template <class Functor, class... Tensors>
void Element(Functor functor, Tensor out, Tensors... tensors);
```

Similarly, [add.inc](file_src_tensors_gpu_add.inc) contains the
specializations for functions matching either of the two signatures in
[add.h](file_src_tensors_gpu_add.h):

```cpp
// src/tensors/gpu/add.h
template <class Functor, class... Tensors>
void Add(Functor functor, float scale, marian::Tensor out, Tensors... tensors);

template <class Functor, class AggFunctor, class... Tensors>
void Aggregate(Functor functor, float initAgg, AggFunctor aggFunctor, float scale, marian::Tensor out, Tensors... tensors);
```

Finally [add_all.inc](file_src_tensors_gpu_add_all.inc) contains the
specializations for [add_all.h](file_src_tensors_gpu_add_all.h), which
are several versions of:

```cpp
// src/tensors/gpu/add_all.h
template <typename T, typename AccType, class Functor, class AggFunctor>
void AggregateAll(Ptr<Allocator> allocator,
                  Functor functor,
                  AccType aggInit,
                  AggFunctor aggFunctor,
                  AccType scale,
                  Tensor out,
                  const Tensor in1);
```

However, for [add_all.h](file_src_tensors_gpu_add_all.h), there is an
additional type dependence in the first template parameter, which requires two
entries:

```cpp
marian::gpu::AggregateAll< float, ... >( ... );
marian::gpu::AggregateAll< __half, ... >( ... );  // for COMPILE_FP16
```

where the `__half` specialization is related to half-precision floats and should
be added to the `COMPILE_FP16` preprocessor block.

The simplest method to add the correct specialization is to take the compilation
error output and extract the needed signature. To extract the signature:

  1. Replace up to, and including, "undefined reference to `" with "template"
  2. Replace the final ' with a semi-colon

To conform with definitions in the codebase, we should replace
`IntrusivePtr<marian::TensorBase>` with its typedef `marian::Tensor`. Note that
as these files are included in `marian::gpu` namespace, and explicitly use
`marian::functional` namespace it is also possible to omit both of these
prefixes. Typically, the namespace prefix of the specialized function is removed
as well. Following these rules for the example of `SinNodeOp` results in the
following entries:

**element**
```cpp
template void Element<Assign<Var<1>, UnaryFunctor<elem::Sin, Assignee<2> > >, marian::Tensor >(Assign<Var<1>, UnaryFunctor<elem::Sin, Assignee<2> > >, marian::Tensor, marian::Tensor);
```

**add**
```cpp
template void Add<BinaryFunctor<elem::Mult,Assignee<1>,UnaryFunctor<elem::Cos,Assignee<2> > >,class marian::Tensor,class marian::Tensor >(BinaryFunctor<elem::Mult,Assignee<1>,UnaryFunctor<elem::Cos,Assignee<2> > >,float,class marian::Tensor,class marian::Tensor,class marian::Tensor);
```

**add_all**
```cpp
template void AggregateAll<float,float,BinaryFunctor<elem::Mult,Assignee<1>,UnaryFunctor<elem::Cos,Assignee<2> > >,BinaryFunctor<elem::Plus,Assignee<1>,Assignee<2> > >(std::shared_ptr<marian::Allocator>,BinaryFunctor<elem::Mult,Assignee<1>,UnaryFunctor<elem::Cos,Assignee<2> > >,float,BinaryFunctor<elem::Plus,Assignee<1>,Assignee<2> >,float,marian::Tensor,marian::Tensor,marian::Tensor);

#if COMPILE_FP16
template void AggregateAll<__half,float,BinaryFunctor<elem::Mult,Assignee<1>,UnaryFunctor<elem::Cos,Assignee<2> > >,BinaryFunctor<elem::Plus,Assignee<1>,Assignee<2> > >(std::shared_ptr<marian::Allocator>,BinaryFunctor<elem::Mult,Assignee<1>,UnaryFunctor<elem::Cos,Assignee<2> > >,float,BinaryFunctor<elem::Plus,Assignee<1>,Assignee<2> >,float,marian::Tensor,marian::Tensor,marian::Tensor);
#endif
```
