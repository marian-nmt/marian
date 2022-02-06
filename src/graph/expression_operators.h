#pragma once
#include "graph/expression_graph.h"
#include "graph/node_initializers.h"

namespace marian {
///@defgroup graph_ops Expression Graph Operators
///@{

/**
 * Assigns a debug message to the expression.
 */
Expr debug(Expr a, const std::string& message = "");

/**
 * Marks the expression as a gradient-checkpoint.
 */
Expr checkpoint(Expr a);

typedef Expr(ActivationFunction)(Expr);  ///< ActivationFunction has signature Expr(Expr)

/**
 * Convenience typedef for graph @ref lambda expressions.
 */
typedef std::function<void(Expr out, const std::vector<Expr>& in)> LambdaNodeFunctor;

/**
 * Arbitrary node with forward operation only.
 */
Expr lambda(const std::vector<Expr>& nodes, Shape shape, Type type, LambdaNodeFunctor fwd, size_t hash = 0);

/**
 * Arbitrary node with forward and backward operation.
 */
Expr lambda(const std::vector<Expr>& nodes, Shape shape, Type type, LambdaNodeFunctor fwd, LambdaNodeFunctor bwd, size_t hash = 0);


/**
 * Convience typedef for graph @ref lambda expressions.
 */
typedef std::function<void(Expr)> LambdaNodeCallback;
Expr callback(Expr node, LambdaNodeCallback call);

/**
 * @addtogroup graph_ops_activation Activation Functions
 * Provides various activation functions for use in the expression.
 * @ingroup graph_ops
 * @{
 */

/**
 * Linear Activation Function.
 * Returns @p nodes[0]
 */
Expr plus(const std::vector<Expr>& nodes);

/**
 * Logistic Activation Function.
 * Computes the <a href="https://en.wikipedia.org/wiki/Logistic_function">logistic function</a>
 * of the given expression
 * @todo rename sigmoid to logistic
 */
Expr sigmoid(Expr a);

/**
 * @copybrief sigmoid
 * @warning not implemented
 */
Expr sigmoid(const std::vector<Expr>& nodes);

/**
 * Swish node.
 * Computes the Swish activation function with @f$\beta=1 @f$
 * @f[
 *    \operatorname{swish}(x) = x \cdot \operatorname{sigmoid}(\beta x)
 * @f]
 * @see SwishNodeOp
 */
Expr swish(Expr a);

/**
 * @copybrief swish
 * @warning not implemented for @p nodes of size > 1
 * @returns swish(nodes[0])
 */
Expr swish(const std::vector<Expr>& nodes);

/**
 * Gaussian Error Linear Unit (GELU).
 * Computes an _approxmiation_ to the Gaussian Error Linear Unit
 * @f[
 *    \operatorname{gelu}(x) = x \cdot \Phi(x)
 *      = x \cdot \frac{1}{2}\left[
 *         1 + \operatorname{erf}\left(\frac{x}{\sqrt{2}}\right)
 *      \right]
 *      \sim \operatorname{swish}(x, 1.702)
 * @f]
 * using @ref SwishNodeOp(a, 1.702)
 * @see SwishNodeOp
 */
Expr gelu(Expr a);

/**
 * @copybrief gelu
 * @warning not implemented for @p nodes of size > 1
 * @returns gelu(nodes[0])
 */
Expr gelu(const std::vector<Expr>&);

/**
 * Tanh.
 * @see TanhNodeOp
 */
Expr tanh(const std::vector<Expr>& nodes);

/**
 * @copybrief tanh
 * Convenience function to put parameter pack @p Args into a Expr vector
 */
template <typename... Args>
Expr tanh(Args... args) {
  std::vector<Expr> nodes{args...};
  return tanh(nodes);
}

/**
 * Rectified Linear Unit (ReLU).
 * Computes the ReLU activation for the Expr
 * @see ReLUNodeOp
 */
Expr relu(Expr a);

/**
 * @copybrief relu
 * @warning not implemented for @p nodes of size > 1
 * @returns relu(nodes[0])
 */
Expr relu(const std::vector<Expr>& nodes);

/**
 * Leaky ReLU (LeakyReLU).
 * Computes the <a href="https://en.wikipedia.org/wiki/Rectifier_(neural_networks)#LeakyReLU">
 * LeakyReLU</a> activation for the expression
 * Activation function:
 * @f[
 *   \operatorname{leakyrelu}(x) =
 *   \begin{cases}
 *     0.01x & \text{if } x \leq 0 \\
 *     x & \text{if } x > 0
 *   \end{cases}
 * @f]
 * @see PReLUNodeOp
 */
Expr leakyrelu(Expr a);

/**
 * @copybrief leakyrelu
 * @warning not implemented
 */
Expr leakyrelu(const std::vector<Expr>& nodes);

/**
 * Parametric Rectified Linear Unit (PReLU).
 * Computes the <a href="https://en.wikipedia.org/wiki/Rectifier_(neural_networks)#Parametric_ReLU">
 * Parametric ReLU</a> activation for the expression
 * @f[
 *   \operatorname{leakyrelu}(x) =
 *   \begin{cases}
 *     \alpha x & \text{if } x \leq 0 \\
 *     x & \text{if } x > 0
 *   \end{cases}
 * @f]
 * @see PReLUNodeOp
 * @note @p alpha is **not** trainable.
 */
Expr prelu(Expr a, float alpha = 0.01);

/**
 * @copybrief prelu
 * @warning not implemented
 */
Expr prelu(const std::vector<Expr>&, float alpha = 0.01);
///@}

/**
 * @addtogroup graph_ops_mathematical Mathematical
 * Performs mathematical operations in the expression graph.
 * @ingroup graph_ops
 * @{
 */

// Exponentiation and Logarithmic functions
/**
 * Natural logarithm.
 * Computes the element-wise natural logarithm of the expression: @f$ \log(a) @f$
 * @see LogNodeOp
 */
Expr log(Expr a);

/**
 * Natural exponentiation.
 * Computes the element-wise natural logarithm of the expression: @f$ e^a @f$
 * @see ExpNodeOp
 */
Expr exp(Expr a);

// Trigonometric functions
/**
* Sine. Computes the element-wise sine of the expression: @f$ \sin(a) @f$.
* @see SinNodeOp
*/
Expr sin(Expr a);

/**
* Cosine. Computes the element-wise cosine of the expression: @f$ \cos(a) @f$.
* @see CosNodeOp
*/
Expr cos(Expr a);

/**
* Tangent. Computes the element-wise tangent of the expression: @f$ \tan(a) @f$.
* @see TanNodeOp
*/
Expr tan(Expr a);
///@}

/**
 * @addtogroup graph_ops_arithmetic Arithmetic
 * Performs arithmetic in the expression graph.
 * @ingroup graph_ops
 * @{
 */

/**
 * Returns @f$ -a @f$.
 * @see NegNodeOp for implementation.
 */
Expr operator-(Expr a);

/*********************************************************/

/**
 * Addition
 * Performs @f$ a + b @f$ in the expression graph.
*/
Expr operator+(Expr a, Expr b);   ///< @see Implementation in PlusNodeOp
Expr operator+(float a, Expr b);  ///< @see Implementation in ScalarAddNodeOp
Expr operator+(Expr a, float b);  ///< @see Implementation in ScalarAddNodeOp

/**
 * Subtraction
 * Performs @f$ a - b @f$ in the expression graph.
 */
Expr operator-(Expr a, Expr b);   ///< @see Implementation in MinusNodeOp
Expr operator-(float a, Expr b);  ///< @see Implementation in ScalarAddNodeOp
Expr operator-(Expr a, float b);  ///< @see Implementation in ScalarAddNodeOp

/**
 * Multiplication
 * Performs @f$ a * b @f$ in the expression graph.
 */
Expr operator*(Expr a, Expr b);   ///< @see Implementation in MultNodeOp
Expr operator*(float a, Expr b);  ///< @see Implementation in ScalarMultNodeOp
Expr operator*(Expr a, float b);  ///< @see Implementation in ScalarMultNodeOp

/**
 * Division
 * Performs @f$ a / b @f$ in the expression graph.
 */
Expr operator/(Expr a, Expr b);   ///< @see Implementation in DivNodeOp
Expr operator/(float a, Expr b);  ///< Promotes @p a to Expression<ConstantNode> and uses operator/(Expr a, Expr b).
                                  ///< @todo efficient version of this without ExpressionGraph::constant
Expr operator/(Expr a, float b);  ///< Implementation via @f$ a * \frac{1}{b} @f$.

///@}

/**
 * Computes the square root of an expression.
 * Evaluates @f$\sqrt{a + \mathrm{eps}} @f$ element-wise on the expression
 * @param a   Expression to square root
 * @param eps Optional positive epsilon to avoid domain errors for small values in @p a
 * @ingroup graph_ops_mathematical
 */
Expr sqrt(Expr a, float eps = 0.f);

/**
 * Computes the square of an expression.
 * Evaluates @f$a^2 @f$ element-wise on the expression
 * @param a Expression to square
 * @ingroup graph_ops_mathematical
 */
Expr square(Expr a);

/**
 * Calculate the element-wise abolute value of an expression.
 * Returns the value of @f$ |a| @f$ element-wise for the expression @p a.
 * @see AbsNodeOp.
 * @ingroup graph_ops_mathematical
 */
Expr abs(Expr a);

// Expr pow(Expr a, Expr b);
// Expr pow(float a, Expr b);
// Expr pow(Expr a, float b);

/**
 * Computes @f$\log(e^a + e^b)@f$.
 */
Expr logaddexp(Expr a, Expr b);


///@addtogroup graph_ops_mathematical
///@{
/*
 * Element-wise min/max
 * Performs an element-wise min max comparison between expressions.
 * @see min, max for axis level operations
 * @see MinimumNodeOp, MaximumNodeOp
 * @todo implement version without ExpressionGraph::constant.
 */

/**
 * Computes the element-wise maximum of its inputs.
 */
Expr maximum(Expr a, Expr b);

/**
 * @copybrief maximum
 * Promotes float input to a @ref ExpressionGraph::constant.
 */
Expr maximum(float a, Expr b);

/**
 * @copybrief maximum
 * Promotes float input to a @ref ExpressionGraph::constant.
 */
Expr maximum(Expr a, float b);

/**
 * Computes the element-wise minimum its inputs.
 */
Expr minimum(Expr a, Expr b);

/**
 * @copybrief minimum
 * Promotes float input to a @ref ExpressionGraph::constant.
 */
Expr minimum(float a, Expr b);

/**
 * @copybrief minimum
 * Promotes float input to a @ref ExpressionGraph::constant.
 */
Expr minimum(Expr a, float b);
///@}

/**
 * Pair of expressions.
 * Currently only used for topk-like nodes
 * @see topk(), argmin(), argmax()
 */
typedef std::tuple<Expr, Expr> Expr2;

/**
 * Pseudo-operator to access elements of a tuple.
 * Provides the same utility as @c std::get<I>(tuple)
 * @see Expr2
 */
template <int I>
Expr get(Expr2 tuple) { return std::get<I>(tuple); }

/**
 * Returns top k elements of an expression along an axis.
 * Return a 2-tuple (values, indices) of the @p k largest, or smallest, elements of an expression
 * along a specified @p axis.
 * The output is ordered according to the value of @p descending.
 * @param a           Expression to search
 * @param k           Number of elements to return
 * @param axis        Axis to along which to operate
 * @param descending  If true, consider the largest elements. Otherwise, consider the smallest elements.
 *                    Default is true.
 * @returns An ordered 2-tuple of Expressions
 */
Expr2 topk(Expr a, int k, int axis, bool descending = true);

/**
 * Returns largest elements of an expression along an axis.
 * Return a 2-tuple (values, indices) of largest elements of an expression
 * along a specified @p axis.
 * @see topk(a, k=1, axis, descending=true)
 */
Expr2 argmax(Expr a, int axis);

/**
 * Returns smallest elements of an expression along an axis.
 * Return a 2-tuple (values, indices) of smallest elements of an expression
 * along a specified @p axis.
 * @see topk(a, k=1, axis, descending=false)
 */
Expr2 argmin(Expr a, int axis);


/**
 * @addtogroup graph_ops_cmp Comparison
 * Performs comparision operations in the expression graph.
 * @ingroup graph_ops
 * Uses CmpNodeOp to perform comparison of graph expression e.g. @f$ a < b @f$.
 * @note
 * We cannot overload the relational operators, as they also mean something for Expr itself.
 * @par
 * @note
 * These names follow <a href="https://pytorch.org/docs">PyTorch</a> convention.
 * @{
 */

/*
 * Expr-Expr comparisons
 */
Expr lt(Expr a, Expr b);  ///< @f$ a < b @f$
Expr eq(Expr a, Expr b);  ///< @f$ a \equiv b @f$
Expr gt(Expr a, Expr b);  ///< @f$ a > b @f$
Expr ge(Expr a, Expr b);  ///< @f$ a \geq b @f$
Expr ne(Expr a, Expr b);  ///< @f$ a \neq b @f$
Expr le(Expr a, Expr b);  ///< @f$ a \leq b @f$

/*
 * Float-Expr comparisons
 * Floats are promoted to a @ref ExpressionGraph::constant and use the Expr-Expr methods
 */
Expr lt(float a, Expr b);  ///< @f$ a < b @f$
Expr eq(float a, Expr b);  ///< @f$ a \equiv b @f$
Expr gt(float a, Expr b);  ///< @f$ a > b @f$
Expr ge(float a, Expr b);  ///< @f$ a \geq b @f$
Expr ne(float a, Expr b);  ///< @f$ a \neq b @f$
Expr le(float a, Expr b);  ///< @f$ a \leq b @f$

Expr lt(Expr a, float b);  ///< @f$ a < b @f$
Expr eq(Expr a, float b);  ///< @f$ a \equiv b @f$
Expr gt(Expr a, float b);  ///< @f$ a > b @f$
Expr ge(Expr a, float b);  ///< @f$ a \geq b @f$
Expr ne(Expr a, float b);  ///< @f$ a \neq b @f$
Expr le(Expr a, float b);  ///< @f$ a \leq b @f$

///@}

/**
 * Computes the dot product of @p a and @p b.
 * Computes @f$ C = \alpha \operatorname{op}(A) \cdot \operatorname{op}(B) @f$,
 * where @f$ \operatorname{op}(A) = A @f$ if @p transA is @c false, and
 * @f$ \operatorname{op}(A) = A^\top @f$ if @c true. The @f$\alpha@f$ parameter
 * is set by @p scalar.
 */
Expr dot(Expr a,
         Expr b,
         bool transA = false,
         bool transB = false,
         float scalar = 1.f);

/**
 * Computes the batch dot product of @p a and @p b.
 * @copydetails dot
 */
Expr bdot(Expr a,
          Expr b,
          bool transA = false,
          bool transB = false,
          float scalar = 1.f);

/**
 * bdot_legacy is an old implemetation of bdot without correct broadcasting on the batch dimensions, 
 * to be removed once the behavior can be correctly replicated with normal bdot on 5 dimensions.
 */
Expr bdot_legacy(Expr a,
                 Expr b,
                 bool transA = false,
                 bool transB = false,
                 float scalar = 1.f);

/**
 * Performs an affine transformation.
 * Computes
 * @f$ C \leftarrow \alpha \operatorname{op}(A) \cdot \operatorname{op}(B) + C@f$,
 * where @f$ \operatorname{op}(A) = A @f$ if @p transA is @c false, and
 * @f$ \operatorname{op}(A) = A^\top @f$ if @c true. The @f$\alpha@f$ parameter
 * is set by @p scalar.
 */
Expr affine(Expr a,
            Expr b,
            Expr bias,
            bool transA = false,
            bool transB = false,
            float scalar = 1.f);

/**
 * As above, but efficiently applies relu transformation to output. For inference only.
 */
Expr affineWithRelu(Expr a,
                    Expr b,
                    Expr bias,
                    bool transA = false,
                    bool transB = false,
                    float scalar = 1.f);

/**
 * Computes the dot product of CSR-tensor @p A with @p B.
 */
Expr csr_dot(const Shape& A_shape, Expr Avalues, Expr Aindices, Expr Aoffsets, Expr B, bool transA = false);

/**
 * Computes the dot product of @p A with CSR-tensor @p B.
 */
Expr dot_csr(Expr A, const Shape& B_shape, Expr B_values, Expr B_indices, Expr B_offsets, bool transB = false);

/**
 * @addtogroup graph_ops_manipulation Manipulation Operations
 * Operators that manipulate expressions.
 * @ingroup graph_ops
 * @{
 */

/**
 * Returns the transpose of an expression.
 * Swaps the last two axes of an expression.
 * @see TransposeNodeOp
 */
Expr transpose(Expr a);

/**
 * Returns the transpose of an expression.
 * Permutes the axes of an expression to resemble @p axes. Axis @c i of the returned
 * expression corresponds to @c axes[i] of the input @p a.
 * @param a     Expression to manipulate
 * @param axes  Desired permutation of axes
 * @see TransposeNodeOp
 */
Expr transpose(Expr a, const std::vector<int>& axes);

/**
 * Swap two axes of an expression.
 * Swaps two axes of an expression via reshaping, if possible, or transpose.
 * @param x      Expression to manipulate
 * @param axis1  Axis to be swapped
 * @param axis2  Axis to swap with
 * @returns Expression with the axes @p axis1 and @p axis2 interchanged
 * @see reshape() and transpose()
 */
Expr swapAxes(Expr x, int axis1, int axis2);

/**
 * Cast an expression to a specified type.
 * @param a     Expression to cast
 * @param type  Desired type
 * @returns     Expression with data cast to @p type
 */
Expr cast(Expr a, Type type = Type::float32);

/**
 * Join a list of expressions along an axis.
 * Concatenates the elements of the expressions in @p concats along the axis @p ax.
 * By default, @p ax operates on the first axis.
 */
Expr concatenate(const std::vector<Expr>& concats, int ax = 0);

/**
 * Repeat elements of an expression.
 * Repeats the elements of @p a along the  @p ax axis @p repeats times.
 * By default, @p ax operates on the first axis.
 * @see concatenate()
 */
Expr repeat(Expr a, size_t repeats, int ax = 0);

/**
 * Reshape expression to a given shape.
 * @param a     The expression to be reshaped
 * @param shape The new shape
 * @returns An expression with shape @p shape.
 */
Expr reshape(Expr a, Shape shape);

/**
 * Clip the values in an expression.
 * Clips the values of the Expr @p a to be within the interval @f$ [-c, c] @f$.
 * @param a Expr to clip
 * @param c Threshold to clip at
 * @see ClipNodeOp
 */
Expr clip(Expr a, float c);

/**
 * Clip the gradient in an expression.
 * Clips the gradient of the Expr @p a to be within the interval @f$ [-c, c] @f$
 * @see clip for the equivalent function which clips values
 * @see ClipGradientNodeOp
 */
Expr clipGradient(Expr a, float c);

/**
 * Converts input to an expression with a least one dimension.
 * @see atleast_nd()
 */
Expr atleast_1d(Expr a);

/**
 * Converts input to an expression with a least two dimensions.
 * @see atleast_nd()
 */
Expr atleast_2d(Expr a);

/**
 * Converts input to an expression with a least three dimensions.
 * @see atleast_nd()
 */
Expr atleast_3d(Expr a);

/**
 * Converts input to an expression with a least four dimensions.
 * @see atleast_nd()
 */
Expr atleast_4d(Expr a);

/**
 * Converts input to an expression with a least n-dimension dimensions.
 * @param a     Expression
 * @param dims  Required number of dimensions
 * @returns     An expression with at least n-dimensions
 */
Expr atleast_nd(Expr a, size_t dims);
///@}

/**
 * @addtogroup graph_ops_creation Creation Operations
 * Operators that create expressions.
 * @ingroup graph_ops
 * @{
 */

/**
 * Create a constant of with the shape of @p a and initialize with @p init.
 * @todo add a && version, to avoid a ref count. NodeInitializers are typically temps.
 * and/or make this a template on init
 */
static inline Expr constant_like(Expr a, const Ptr<inits::NodeInitializer>& init) {
  return a->graph()->constant(a->shape(), init, a->value_type());
}

/**
 * Convenience function to initialize from a vector.
 */
template<typename ElementType>
Expr constant_like(Expr a, const std::vector<ElementType>& v) { return constant_like(a, inits::fromVector(std::move(v))); }

/**
 * Convenience function to initialize from a vector.
 */
template<typename ElementType>
Expr constant_like(Expr a, std::vector<ElementType>&& v) { return constant_like(a, inits::fromVector(v)); }

///@}

/**
 * @addtogroup graph_ops_manipulation
 * @{
 */

/**
 * Flattens an expression to one dimension.
 * @see ReshapeNodeOp
 */
Expr flatten(Expr a);

/**
 * Flattens an expression to two-dimensions preserving the last dimension.
 * @see ReshapeNodeOp
 */
Expr flatten_2d(Expr a);

///@}

/**
 * Wraps an expression as a non-trainable expression.
 */
Expr stopGradient(Expr a);

/**
 * Gathers elements along an axis.
 * @param a       The input expression
 * @param axis    The axis along which to index
 * @param indices The indices to be gathered
 * @returns       Gathered expression with the same shape as @p indices
 * @note @p a and @p indices must have the same rank
 * @note The non-target axes of @p a and @p indices must have the same size, or be broadcastable.
 */
Expr gather(Expr a, int axis, Expr indices);

/**
 * Scatter elements from source along an axis into a. Unindexed elements from a remain unchanged.
 * This is the reverse operation to gather.
 * @param a       The input expression
 * @param axis    The axis along which to index
 * @param indices The indices to be scattered
 * @param source  Expression with values to scatter. 
 * @returns       Scattered expression with the same shape as @p a now containing values from @p source in positions @p indices
 * @note @p source and @p indices must have the same rank
 * @note In this version @p source and @p indicies must have the same shape
 */
Expr scatter(Expr a, int axis, Expr indices, Expr source);

#if 0
 // reverse operation to gather. a is expression into with values from b are inserted and positions indices along axis.
 // with broadcasting

 auto knn = get<0>(KNN->apply(query, k)); // [beam, time, batch, k]

 auto W = reshape(gather(Wt_, -2, flatten(knn)), {beam * time * batch, k, dim});
 auto b = reshape(gather(b_,  -1, flatten(knn)), {beam * time * batch, 1, k });
 query       = reshape(query, {beam * time * batch, 1, dim});
 auto logits = bdot(query, W, false, true); // [beam * time * batch, 1, k]
 logits      = reshape(logits + b, {beam, time, batch, k}); // @TODO: add baffine node

 auto shape = indices.shape();
 shape.set(-1, 32000);
 auto output = grep->constant(shape, inits::lowest(), logits->value_type());
 output = scatter(output, -1, indices, logits);

 // auto a = graph->constant({2,2,5,32000}, inits::fromValue(minimal))
 // scatter(a, -1, indices, values)
 // PyTorch does for out-of-place scatter: out = a.scatter(-1, indices, values)
Expr scatter(Expr a, int axis, Expr indices, Expr b);

#endif

/**
 * Returns a new expression containing the @p indicies of expression @p a
 * along the specified @p axis.
 * @warning Do not pass a scalar literal 0 as @p indices;
 * it will compile but pass a nullptr.
 */
Expr index_select(Expr a, int axis, Expr indices);

/**
 * @copybrief index_select
 * Convenience wrapper that promotes a vector of @ref IndexType to an Expr
 */
Expr index_select(Expr a, int axis, const std::vector<IndexType>& indices);

/**
 * Performs an @ref index_select() along the first axis.
 * @see index_select()
 */
static inline Expr rows(Expr a, Expr indices) {
  return index_select(a, 0, indices);
}

/**
 * @copybrief rows
 * Convenience wrapper that promotes a vector of @ref IndexType to an Expr
 */
static inline Expr rows(Expr a, const std::vector<IndexType>& indexVector) {
  return index_select(a, 0, indexVector);
}

/**
 * Performs an @ref index_select() along the last axis.
 * @see index_select()
 */
static inline Expr cols(Expr a, Expr indices) {
  return index_select(a, -1, indices);
}

/**
 * @copybrief cols
 * Convenience wrapper that promotes a vector of @ref IndexType to an Expr
 */
static inline Expr cols(Expr a, const std::vector<IndexType>& indexVector) {
  return index_select(a, -1, indexVector);
}

/**
 * Returns the @p slice of the expression @p a along @p axis.
 * @see Slice
 */
Expr slice(Expr a, int axis, Slice slice);

/**
 * @copybrief slice
 * Convenience wrapper for slice() that returns the slice along @p axis
 * from @p index to @p index+1
 */
static inline Expr slice(Expr a, int axis, int index) {
  return slice(a, axis, Slice(index));
}

/**
 * @copybrief slice
 * Convenience wrapper for slice() that returns the slice along @p axis
 * from @p index to @p index + @p length
 * @note this is named after an equivalent function in PyTorch
 */
static inline Expr narrow(Expr a, int axis, size_t start, size_t length) {
  return slice(a, axis, Slice((int)start, (int)(start + length)));
}

/*********************************************************/

///@addtogroup graph_ops_mathematical
///@{
// Aggregations

/**
 * Compute the sum along the specified axis.
 * @param ax Axis along which to compute the sum. Default is @c 0.
 * @see ReduceNodeOp
 */
Expr sum(Expr a, int ax = 0);

/**
 * Compute the arithmetic mean along the specified axis.
 * @param ax Axis along which to compute the mean. Default is @c 0.
 * @see ReduceNodeOp
 */
Expr mean(Expr a, int ax = 0);

/**
 * Compute the standard deviation along the specified axis.
 * @param ax Axis along which to compute the standard deviation
 * @see ReduceNodeOp
 */
Expr std(Expr a, int ax);

/**
 * Compute the variance along the specified axis.
 * @param ax Axis along which to compute the variance
 * @see ReduceNodeOp
 */
Expr var(Expr a, int ax);

/**
 * Compute the maximum along the specified axis.
 * @param ax Axis along which to find the maximum
 * @see ReduceNodeOp
 */
Expr max(Expr a, int ax);

/**
 * Compute the minimum along the specified axis.
 * @param ax Axis along which to find the minimum
 * @see ReduceNodeOp
 */
Expr min(Expr a, int ax);

/**
 * Compute the product along the specified axis.
 * @param ax Axis along which to compute the product
 * @see ReduceNodeOp
 */
Expr prod(Expr a, int ax);

///@}

/**
 * Compute the log of the sum of exponentials along the specified axis.
 * @param ax Axis along which to perform the operation
 * @see ReduceNodeOp
 */
Expr logsumexp(Expr a, int ax);

/**
 * Computes the softmax fuction along the given axis.
 * Applies the softmax function
 * @f[
    \operatorname{softmax}(x_i) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}
 * @f]
 * @see SoftmaxNodeOp
 */
Expr softmax(Expr x, int axis = -1);

/**
 * @copybrief softmax
 * Applies the softmax function over the unmasked values.
 * @see SoftmaxNodeOp
 */
Expr softmax(Expr a, Expr zeroOneMask, int axis = -1);

/**
 * Computes the log of the softmax function along the last axis.
 * Applies @f$ \log(\operatorname{softmax}(x)) @f$.
 * @see LogSoftmaxNodeOp
 */
Expr logsoftmax(Expr a);

/**
 * Computes the cross-entropy loss.
 * @param labelSmoothingAlpha The amount of label smoothing @f$\alpha \in [0,1]@f$.
 * Default is no smoothing, @f$\alpha = 0 @f$.
 * @see CrossEntropyNodeOp
 */
Expr cross_entropy(Expr a, Expr b, float labelSmoothingAlpha = 0.f, Type outputType = Type::float32);

/**
 * Computes the unlikelihood loss.
 * Computes the <a href="https://arxiv.org/abs/1908.04319">unlikelihood</a> loss
 * @f$ -\log \sum (1 - \operatorname{softmax}(x)) @f$
 */
Expr unlikelihood(Expr a, Expr b);

/**
 * Computes the scalar product along the specified axis.
 * @see ScalarProductNodeOp
 */
Expr scalar_product(Expr a, Expr b, int ax = 0);

/**
 * Compute the weighted arithmetic mean along the specified axis.
 */
Expr weighted_average(Expr in, Expr weights, int ax = 0);


/**
 * Applies layer normalization over the last dimension.
 * @f[
   \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \mathrm{eps}}} \times \gamma + \beta
 * @f]
 * @see LayerNormalizationOp
 */
Expr layerNorm(Expr x, Expr gamma, Expr beta = nullptr, float eps = 1e-9);

/**
 * Applies RMS normalization over the last dimension. 
 * 
 * See: Biao Zhang; Rico Sennrich (2019). Root Mean Square Layer Normalization. 
 * In Advances in Neural Information Processing Systems 32. Vancouver, Canada.
 * @f[
   \frac{x}{\sqrt{\frac{1}{N}\sum x^2 + \mathrm{eps}}} \times \gamma + \beta
 * @f]
 * @see RMSNormalizationOp
 */
Expr rmsNorm(Expr x, Expr gamma, Expr beta = nullptr, float eps = 1e-9);

/**
 * Highway transformation.
 * Computes the highway tranform on @p y and @p x as gated by @p t:
 * @f$ \operatorname{sigmoid}(t) y + (1-\operatorname{sigmoid}(t)) x @f$
 * @see HighwayNodeOp
 */
Expr highway(Expr y, Expr x, Expr t);

/** @copybrief highway
 * Generates a highway network for @p x with a @ref relu activated layer and
 * @ref sigmoid activated layer for gating.
 * @see mlp::dense()
 */
Expr highway(const std::string prefix, Expr x);

/**
 * Performs dropout using a given mask.
 */
static inline Expr dropout(Expr x, Expr mask) {
  if (mask)
    return x * mask;
  else
    return x;
}

/**
 * Performs dropout with a given probably and explicit shape.
 */
static inline Expr dropout(Expr x, float dropProb, Shape shape) {
  if(dropProb == 0)
    return x;
  auto graph = x->graph();
  auto mask = graph->dropoutMask(dropProb, shape);
  return dropout(x, mask);
}

/**
 * Performs dropout with a given probably.
 */
static inline Expr dropout(Expr x, float dropProb) {
  if(dropProb == 0)
    return x;
  return dropout(x, dropProb, x->shape());
}

/**
 * Shifts the elements of an expression by a per-axis offset @p shift
 * padded with @p padValue.
 */
Expr shift(Expr x, Shape shift, float padValue = 0);

/**
 * Reindexes an expression from internal to cuDNN format.
 */
Expr convert2cudnnFormat(Expr x);

/**
 * Reindexes an expression from cuDNN to internal format.
 */
Expr convertFromcudnnFormat(Expr x);

/**
 * Performs average pooling.
 * @see PoolingOp
 */
Expr avg_pooling(Expr x,
                 int height,
                 int width,
                 int padHeight = 0,
                 int padWidth = 0,
                 int strideHeight = 1,
                 int strideWidth = 1);

/**
 * Performs max pooling.
 * @see PoolingOp
 */
Expr max_pooling(Expr x,
                 int height,
                 int width,
                 int padHeight = 0,
                 int padWidth = 0,
                 int strideHeight = 1,
                 int strideWidth = 1);

/**
 * Pooling operation with masking.
 * @warning not implemented
 */
Expr pooling_with_masking(Expr x, Expr mask, int width, bool isEven = false);

///@}
}  // namespace marian
