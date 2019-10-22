#include "graph/expression_operators.h"
#include "layers/constructors.h"

#include "graph/node_operators.h"
#include "graph/node_operators_binary.h"
#include "graph/node_operators_unary.h"

#include "graph/auto_tuner.h"
#include "tensors/cpu/int16.h"
#include "tensors/cpu/expanded_gemm.h"

#if USE_FBGEMM
#include "fbgemm/Utils.h"
#endif

namespace marian {

Expr debug(Expr a, const std::string& message) {
  a->debug(message);
  return a;
}

Expr checkpoint(Expr a) {
  a->markCheckpoint();
  return a;
}

// logistic function. Note: scipy name is expit()
Expr sigmoid(Expr a) {
  return Expression<SigmoidNodeOp>(a);
}

Expr relu(Expr a) {
  return Expression<ReLUNodeOp>(a);
}

Expr leakyrelu(Expr a) {
  return Expression<PReLUNodeOp>(0.01f, a);
}

Expr prelu(Expr a, float alpha) {
  return Expression<PReLUNodeOp>(alpha, a);
}

Expr clip(Expr a, float c) {
  if(c == 0)
    return a;
  else
    return Expression<ClipNodeOp>(a, c);
}

Expr log(Expr a) {
  return Expression<LogNodeOp>(a);
};

Expr exp(Expr a) {
  return Expression<ExpNodeOp>(a);
};

Expr swish(Expr a) {
  return Expression<SwishNodeOp>(a);
}

Expr gelu(Expr a) {
  return Expression<SwishNodeOp>(a, 1.702f);
}

Expr operator-(Expr a) {
  return Expression<NegNodeOp>(a);
};

Expr softmax(Expr a, int axis /*=-1*/)
{
  // @TODO: move axis parameter down into the kernel
  if (axis != -1)
  {
    return swapAxes(softmax(swapAxes(a,
                                     axis, -1),
                            /*axis=*/-1),
                    axis, -1);
  }
  return Expression<SoftmaxNodeOp>(a);
}

Expr softmax(Expr a, Expr zeroOneMask, int axis /*=-1*/) {
  // This will return the smallest value / 2 for the input type converted to float
  // So for Type::Float16 that will be the smallest fp16 value expressed as float
  // We divide by 2 to allow for some tolerance and overflow protection.
  float smallestFloat = NumericLimits<float>(a->value_type()).lowest / 2.f;
  auto logMask = (1.f - zeroOneMask) * smallestFloat;
  return softmax(a + logMask, axis);
}

// @TODO: add mask
Expr logsoftmax(Expr a) {
  return Expression<LogSoftmaxNodeOp>(a);
}

/*********************************************************/

Expr operator+(Expr a, Expr b) {
  return Expression<PlusNodeOp>(a, b);
}

Expr operator-(Expr a, Expr b) {
  return Expression<MinusNodeOp>(a, b);
}

Expr operator*(Expr a, Expr b) {
  return Expression<MultNodeOp>(a, b);
}

Expr operator/(Expr a, Expr b) {
  return Expression<DivNodeOp>(a, b);
}

Expr logaddexp(Expr a, Expr b) {
  return Expression<LogAddExpNodeOp>(a, b);
}

Expr maximum(Expr a, Expr b) {
  return Expression<MaximumNodeOp>(a, b);
}

Expr minimum(Expr a, Expr b) {
  return Expression<MinimumNodeOp>(a, b);
}

Expr lt(Expr a, Expr b) { return Expression<CmpNodeOp>(a, b, -1, false); }
Expr eq(Expr a, Expr b) { return Expression<CmpNodeOp>(a, b,  0, false); }
Expr gt(Expr a, Expr b) { return Expression<CmpNodeOp>(a, b,  1, false); }
Expr ge(Expr a, Expr b) { return Expression<CmpNodeOp>(a, b, -1,  true); }
Expr ne(Expr a, Expr b) { return Expression<CmpNodeOp>(a, b,  0,  true); }
Expr le(Expr a, Expr b) { return Expression<CmpNodeOp>(a, b,  1,  true); }

Expr lt(float a, Expr b) { return Expression<CmpNodeOp>(b->graph()->constant({}, inits::fromValue(a), b->value_type()), b, -1, false); }
Expr eq(float a, Expr b) { return Expression<CmpNodeOp>(b->graph()->constant({}, inits::fromValue(a), b->value_type()), b,  0, false); }
Expr gt(float a, Expr b) { return Expression<CmpNodeOp>(b->graph()->constant({}, inits::fromValue(a), b->value_type()), b,  1, false); }
Expr ge(float a, Expr b) { return Expression<CmpNodeOp>(b->graph()->constant({}, inits::fromValue(a), b->value_type()), b, -1,  true); }
Expr ne(float a, Expr b) { return Expression<CmpNodeOp>(b->graph()->constant({}, inits::fromValue(a), b->value_type()), b,  0,  true); }
Expr le(float a, Expr b) { return Expression<CmpNodeOp>(b->graph()->constant({}, inits::fromValue(a), b->value_type()), b,  1,  true); }

Expr lt(Expr a, float b) { return Expression<CmpNodeOp>(a, a->graph()->constant({}, inits::fromValue(b), a->value_type()), -1, false); }
Expr eq(Expr a, float b) { return Expression<CmpNodeOp>(a, a->graph()->constant({}, inits::fromValue(b), a->value_type()),  0, false); }
Expr gt(Expr a, float b) { return Expression<CmpNodeOp>(a, a->graph()->constant({}, inits::fromValue(b), a->value_type()),  1, false); }
Expr ge(Expr a, float b) { return Expression<CmpNodeOp>(a, a->graph()->constant({}, inits::fromValue(b), a->value_type()), -1,  true); }
Expr ne(Expr a, float b) { return Expression<CmpNodeOp>(a, a->graph()->constant({}, inits::fromValue(b), a->value_type()),  0,  true); }
Expr le(Expr a, float b) { return Expression<CmpNodeOp>(a, a->graph()->constant({}, inits::fromValue(b), a->value_type()),  1,  true); }

/*********************************************************/

Expr operator+(Expr a, float b) {
  if (b == 0)
    return a;
  else
    return Expression<ScalarAddNodeOp>(a, b);
}

Expr operator+(float a, Expr b) {
  if (a == 0)
    return b;
  else
    return Expression<ScalarAddNodeOp>(b, a);
}

Expr operator-(Expr a, float b) {
  if (b == 0)
    return a;
  else
    return Expression<ScalarAddNodeOp>(a, -b);
}

Expr operator-(float a, Expr b) {
  if (a == 0)
    return -b;
  else
    return Expression<ScalarAddNodeOp>(-b, a);
}

Expr operator*(float a, Expr b) {
  if (a == 1.0f)
    return b;
  else
    return Expression<ScalarMultNodeOp>(b, a);
}

Expr operator*(Expr a, float b) {
  if (b == 1.0f)
    return a;
  else
    return Expression<ScalarMultNodeOp>(a, b);
}

Expr operator/(Expr a, float b) {
  return a * (1.f / b);
}

// TODO: efficient version of this without constant()
Expr operator/(float a, Expr b) {
  auto aExpr = b->graph()->constant({}, inits::fromValue(a));
  return aExpr / b;
}

// Expr pow(float a, Expr b) {
//  return Expression<Scalar1PowNodeOp>(a, b);
//
//}
//
// Expr pow(Expr a, float b) {
//  return Expression<Scalar2PowNodeOp>(a, b);
//
//}
//
// Expr pow(Expr a, Expr b) {
//  return Expression<PowNodeOp>(a, b);
//}

/*********************************************************/

Expr concatenate(const std::vector<Expr>& concats, int ax) {
  return Expression<ConcatenateNodeOp>(concats, ax);
}

Expr repeat(Expr a, size_t repeats, int ax) {
  if(repeats == 1)
    return a;
  return concatenate(std::vector<Expr>(repeats, a), ax);
}

Expr reshape(Expr a, Shape shape) {
  if (a->shape() == shape)
    return a;
  return Expression<ReshapeNodeOp>(a, shape);
}

// @TODO: remove this if it turns out that we can train FP16 without that
Expr clipGradient(Expr a, float clipValue) {
  // don't create node if no clipping
  return clipValue != 0.f ? Expression<ClipGradientNodeOp>(a, clipValue) : a;
}

Expr atleast_1d(Expr a) {
  return atleast_nd(a, 1);
}

Expr atleast_2d(Expr a) {
  return atleast_nd(a, 2);
}

Expr atleast_3d(Expr a) {
  return atleast_nd(a, 3);
}

Expr atleast_4d(Expr a) {
  return atleast_nd(a, 4);
}

Expr atleast_nd(Expr a, size_t dims) {
  if(a->shape().size() >= dims)
    return a;

  Shape nShape;
  nShape.resize(dims);
  for(int i = 1; i <= (int)a->shape().size(); ++i)
    nShape.set(-i, a->shape()[-i]);

  return reshape(a, nShape);
}

Expr flatten(Expr a) {
  Shape shape = {a->shape().elements()};
  return Expression<ReshapeNodeOp>(a, shape);
}

Expr flatten_2d(Expr a) {
  Shape shape = {a->shape().elements() / a->shape()[-1], a->shape()[-1]};
  return Expression<ReshapeNodeOp>(a, shape);
}

Expr stopGradient(Expr a) {
  // implemented as a dummy reshape that is not trainable
  auto res = Expression<ReshapeNodeOp>(a, a->shape());
  res->setTrainable(false);
  return res;
}

Expr constant_like(Expr a, const Ptr<inits::NodeInitializer>& init) {
  auto graph = a->graph();
  return graph->constant(a->shape(), init, a->value_type());
}

// gather() -- gather arbitrary elements along an axis; batched or non-batched
Expr gather(Expr a, int axis, Expr indices) {
  return Expression<GatherNodeOp>(a, axis, indices);
}

// index_select() -- gather arbitrary elements along an axis from an unbatched
// input 'a'. Indices are specified as a 1D vector.
// This is used e.g. for embedding lookup.
// Note: To use a batch of index vectors, reshape them into a single vector,
// call index_select(), then reshape the result back. Reshapes are cheap.
// This function has the same semantics as PyTorch operation of the same name.
Expr index_select(Expr a, int axis, Expr indices) {
  ABORT_IF(indices->shape().size() != 1, "Indices must be a 1D tensor");
  // We have specialized kernels for non-batched indexing of first or last axis of a 2D tensor.
  auto rank = a->shape().size();
  if (rank == 2) {
    if (axis == 0 || axis == -2)
      return Expression<RowsNodeOp>(a, indices);
    else if (axis == -1 || axis == 1)
      return Expression<ColsNodeOp>(a, indices);
  }
  // Delegate to gather() for any other axis or non-matrix input.
  Shape shape;
  shape.resize(a->shape().size());
  shape.set(axis, indices->shape()[0]);
  indices = reshape(indices, shape); // move index to axis
  return gather(a, axis, indices);
}
Expr index_select(Expr a, int axis, const std::vector<IndexType>& indices) {
  auto indexExpr = a->graph()->indices(indices);
  return index_select(a, axis, indexExpr);
}

static Expr sliceCopy(Expr a, int axis, const Slice& slice) { // copy a Slice via gather()
  ABORT_IF(slice.stride < 0, "Negative strides are not supported yet");
  ABORT_IF(slice.begin == slice.end, "Empty slices are not allowed"); // @TODO: Or are they?
  std::vector<IndexType> indices;
  indices.reserve((slice.end - slice.begin - 1) / slice.stride + 1);
  for (int i = slice.begin; i < slice.end; i += slice.stride)
    indices.push_back((IndexType)i);
  return gather(a, axis, a->graph()->indices(indices, a, axis));
}

static Expr sliceView(Expr a, int axis, const Slice& slice) { // view a slice (must be memory-consecutive)
  return Expression<SliceViewNodeOp>(a, axis, slice);
}

// slice() -- gather a slice along an axis (step size > 1 allowed)
Expr slice(Expr a, int axis, Slice slice) { // numpy __getslice__ semantics, but with axis parameter
  const auto& shape = a->shape();
  axis  = shape.axis(axis);         // normalize negative axis
  slice = shape.slice(slice, axis); // normalize negative slice values
  if (slice.begin == 0 && slice.end == shape[axis] && slice.stride == 1)
    return a; // it's a no-op
#if 1 // until strided views are supported, non-consecutive slices are implemented via gather()
  if (slice.stride != 1)
    return sliceCopy(a, axis, slice);
  for (int i = 0; i < axis; ++i) {
    if (shape[i] != 1)  // this makes it non-consecutive
      return sliceCopy(a, axis, slice);
  }
#endif
  return sliceView(a, axis, slice);
}

Expr sum(Expr a, int ax) {
  return Expression<ReduceNodeOp>(a, ax, ReduceNodeOpCode::sum);
}

Expr mean(Expr a, int ax) {
  return Expression<ReduceNodeOp>(a, ax, ReduceNodeOpCode::mean);
}

Expr std(Expr a, int ax) {
  return Expression<ReduceNodeOp>(a - mean(a,ax), ax, ReduceNodeOpCode::rms);
}

Expr var(Expr a, int ax) {
  return Expression<ReduceNodeOp>(a - mean(a, ax), ax, ReduceNodeOpCode::meanSqr);
}

Expr max(Expr a, int ax) {
  return Expression<ReduceNodeOp>(a, ax, ReduceNodeOpCode::max);
}

Expr min(Expr a, int ax) {
  return Expression<ReduceNodeOp>(a, ax, ReduceNodeOpCode::min);
}

Expr prod(Expr a, int ax) {
  return Expression<ReduceNodeOp>(a, ax, ReduceNodeOpCode::prod);
}

// log(sum(exp(a)))
Expr logsumexp(Expr a, int ax) {
  return Expression<ReduceNodeOp>(a, ax, ReduceNodeOpCode::logSumExp);
}

Expr scalar_product(Expr a, Expr b, int ax) {
  return Expression<ScalarProductNodeOp>(a, b, ax);
}

Expr weighted_average(Expr in, Expr weights, int ax) {
  auto p = scalar_product(in, weights, ax);
  auto s = sum(weights, ax);
  return p / s;
}

Expr dot(Expr a, Expr b, bool transA, bool transB, float scale) {
  auto device = a->graph()->getDeviceId().type;
  float clipValue = a->graph()->getBackend()->getClip();

  // Currently only true when command line options
  // --optimize --cpu-thread=N with N > 0 are set.
  if(device == DeviceType::cpu && a->graph()->getBackend()->isOptimized()
     && a->graph()->getBackend()->getGemmType() == GemmType::IntrinInt16) {
    // dotInt16 computes A * B.T, hence the transpose for B to get A * B
    // if transA = false and transB = false.

    return cpu::int16::dot(
        cpu::int16::quantize(transA ? transpose(a) : a, clipValue),
        cpu::int16::quantize(transB ? b : transpose(b), clipValue),
        scale);
  } else {
    return Expression<DotNodeOp>(
        clip(a, clipValue), clip(b, clipValue), transA, transB, scale);
  }
}

Expr bdot(Expr a, Expr b, bool transA, bool transB, float scale) {
  return Expression<DotBatchedNodeOp>(a, b, transA, transB, scale);
}

Expr affine(Expr a, Expr b, Expr bias, bool transA, bool transB, float scale) {
  auto device = a->graph()->getDeviceId().type;

  float clipValue = a->graph()->getBackend()->getClip();

  if(device == DeviceType::cpu && a->graph()->getBackend()->isOptimized()) {
    GemmType gemmType = a->graph()->getBackend()->getGemmType();
    // When gemmType is set to 'auto', an autotuner decides the best algorithm available.
    // A new autotuner is created, then different kinds of algorithms are added to the autotuner.
    // For each GEMM size, there is a unique hash key.
    // (e.g. m, n, k, transpose A, transpose B, bias size for GEMM)
    if(gemmType == GemmType::Auto) {
      thread_local Ptr<AutoTuner<Expr>> tuner = New<AutoTuner<Expr>>();

      // start with new set of algorithms
      tuner->clear();

      // lower precicion for shapes, reduces data sparsity
      auto sh = [](Shape sh) {
        for(size_t i = 0; i < sh.size(); ++i)
          sh.set(i, sh[i] / 4);
        return sh;
      };

      // create context for current call as hash
      std::size_t hash = sh(a->shape()).hash();
      util::hash_combine(hash, sh(b->shape()).hash());
      util::hash_combine(hash, sh(bias->shape()).hash());
      util::hash_combine(hash, transA);
      util::hash_combine(hash, transB);

#if USE_FBGEMM
      // Use Packed GEMM only if the node b in the graph is memoized.
      // More specifically, packed GEMM is used only if the B matrix (weight) is constant.
      // In general, 'memoized' means that the node is a constant variable or
      // a combination of contant nodes which is also a constant variable
      // when it's computed once.
      // Those memoized nodes are cached to avoid duplicated computations.
      // 07/10/2019 - Use packed GEMM only if the cpu architecture supports AVX2
      // one of the fbgemm's sub modules, cpuinfo (https://github.com/pytorch/cpuinfo).
      // It looks at the cpu register 
      // (https://github.com/pytorch/cpuinfo/blob/master/src/x86/isa.c#L391),
      // and this cpu lookup is executed only once and the state is kept in FBGEMM.
      if(fbgemm::fbgemmHasAvx2Support() && b->memoize()) {
        // add packed GEMM algorithm variant (Packed GEMM) to the autotuner
        // Once an algorithm is added to the autotuner,
        // autotuner runs all the added algorithms for a designated times.
        // One algorithm is run per one this operation call
        // and the stat for that algorithm is collected.
        // When all the algorithms reach the maximum stat collection count,
        // the autotuner decide the best algorithm, and keep using it afterward.
        size_t hashPack = hash;
        util::hash_combine(hashPack, 1);
        auto recPack = [=](Expr e, bool stop = false) {
          e->record(tuner, hashPack, stop);
          return e;
        };

        auto algPack = [=]() {
          auto packed = cpu::variant::pack(b, cpu::variant::PackMatrix::B, transB, clipValue);

          return recPack(
              cpu::variant::affine(
                  clip(a, clipValue),
                  packed,
                  b->shape(),
                  bias,
                  transA,
                  transB,
                  scale),
              true);
        };
        tuner->insert({hashPack, algPack});
      }
#endif // USE_FBGEMM

      // add second algorithm variant (Int16) to the autotuner
      size_t hashInt16 = hash;
      util::hash_combine(hashInt16, 2);
      auto recInt16 = [=](Expr e, bool stop = false) {
        e->record(tuner, hashInt16, stop);
        return e;
      };
      auto algInt16 = [=]() {
        return recInt16(
            cpu::int16::affine(
                recInt16(
                    cpu::int16::quantize(
                        transA ? recInt16(transpose(a)) : a,
                        clipValue)),
                cpu::int16::quantize(
                    transB ? b : transpose(b),
                    clipValue),
                bias,
                scale),
            true);
      };
      tuner->insert({hashInt16, algInt16});

      // add third algorithm variant (CBlas) to the autotuner
      size_t hashCblas = hash;
      util::hash_combine(hashCblas, 3);
      auto recCblas = [=](Expr e, bool stop = false) {
        e->record(tuner, hashCblas, stop);
        return e;
      };

      auto algCblas = [=]() {
        auto ac = clip(a, clipValue);
        if(ac != a)
          ac = recCblas(ac);

        auto bc = clip(b, clipValue);
        if(bc != b)
          bc = recCblas(bc);

        int rows = ac->shape().elements() / ac->shape()[-1];
        Expr ones = ac->graph()->ones({rows, 1}, bias->value_type());
        std::vector<Expr> nodes = {ac, bc, bias, ones};
        return recCblas(Expression<AffineNodeOp>(nodes, transA, transB, scale),
                        true);
      };
      tuner->insert({hashCblas, algCblas});

      // execute algorithm with autotuning
      return tuner->run();
    } else {
      if(gemmType == GemmType::IntrinInt16) {
        // cpu int16 version
        return cpu::int16::affine(
            cpu::int16::quantize(transA ? transpose(a) : a, clipValue),
            cpu::int16::quantize(transB ? b : transpose(b), clipValue),
            bias,
            scale);
      } else if(gemmType == GemmType::FbFp16Packed) {
#if USE_FBGEMM
        // 07/10/2019 - Use packed GEMM only if the cpu architecture supports AVX2
        // one of the fbgemm's sub modules, cpuinfo (https://github.com/pytorch/cpuinfo).
        // It looks at the cpu register
        // (https://github.com/pytorch/cpuinfo/blob/master/src/x86/isa.c#L391),
        // and this cpu lookup is executed only once and the state is kept in FBGEMM.
        if(fbgemm::fbgemmHasAvx2Support() && b->memoize()) {
          auto packed = cpu::variant::pack(b, cpu::variant::PackMatrix::B, transB, clipValue);

          return cpu::variant::affine(
              clip(a, clipValue),
              packed,
              b->shape(),
              bias,
              transA,
              transB,
              scale);
        } else {
          int rows = a->shape().elements() / a->shape()[-1];
          Expr ones = a->graph()->ones({rows, 1}, bias->value_type());
          std::vector<Expr> nodes = {clip(a, clipValue), clip(b, clipValue), bias, ones};
          return Expression<AffineNodeOp>(nodes, transA, transB, scale);
        }
#else
        ABORT("Packed GEMM is not available in this build");
#endif  // USE_FBGEMM

      } else if(gemmType == GemmType::MklFp32) {
        // general version, MKL, CBlas or CUDA

        // if clipValue > 0, the inputs will be clipped to range [-clipValue,
        // clipValue] This is meant to keep values at the same range as used during
        // training when optimizing for 8-bit integer products. Likely to be removed
        // in the future when we explore better ways to handle this.

        int rows = a->shape().elements() / a->shape()[-1];
        Expr ones = a->graph()->ones({rows, 1}, bias->value_type());
        std::vector<Expr> nodes
            = {clip(a, clipValue), clip(b, clipValue), bias, ones};
        return Expression<AffineNodeOp>(nodes, transA, transB, scale);
      } else {
        ABORT("GemmType..{} not available by affine()", gemmType);
      }
    }
  } else {
    // general version, MKL, CBlas or CUDA

    // if clipValue > 0, the inputs will be clipped to range [-clipValue,
    // clipValue] This is meant to keep values at the same range as used during
    // training when optimizing for 8-bit integer products. Likely to be removed
    // in the future when we explore better ways to handle this.

    int rows = a->shape().elements() / a->shape()[-1];
    Expr ones = a->graph()->ones({rows, 1});
    std::vector<Expr> nodes
        = {clip(a, clipValue), clip(b, clipValue), bias, ones};
    return Expression<AffineNodeOp>(nodes, transA, transB, scale);
  }
}

// multiply a CSR matrix A with a matrix B
// A[i,j] is at A_values[A_offsets[i]+k], where k is position of j in A_indices[A_offsets[i]:A_offsets[i+1]]
// @TODO: Define a proper sparse tensor type.
Expr csr_dot(const Shape& A_shape, Expr A_values, Expr A_indices, Expr A_offsets, Expr B, bool transA /*= false*/) {
  return Expression<CSRDotNodeOp>(A_shape, A_values, A_indices, A_offsets, B, transA, /*swapOperands=*/false);
}

// multiply a matrix A with a CSR matrix B
// @TODO: Define a proper sparse tensor type.
Expr dot_csr(Expr A, const Shape& B_shape, Expr B_values, Expr B_indices, Expr B_offsets, bool transB /*= false*/) {
  return Expression<CSRDotNodeOp>(B_shape, B_values, B_indices, B_offsets, A, transB, /*swapOperands=*/true);
}

// swap the last two axes
// @TODO: change to swapAxes(a, -1, -2)
Expr transpose(Expr a) {
  std::vector<int> axes(a->shape().size());
  for(int i = 0; i < axes.size(); ++i) {
    axes[i] = i;
  }
  if(axes.size() > 1) {
    axes[axes.size() - 1] = (int)axes.size() - 2;
    axes[axes.size() - 2] = (int)axes.size() - 1;
  }
  return Expression<TransposeNodeOp>(a, axes);
}

Expr transpose(Expr a, const std::vector<int>& axes) {
  return Expression<TransposeNodeOp>(a, axes);
}

Expr swapAxes(Expr x, int axis1, int axis2)
{
  const auto& shape = x->shape();
  axis1 = shape.axis(axis1);
  axis2 = shape.axis(axis2);
  if (axis1 == axis2)
    return x;
  if (shape[axis1] == 1 || shape[axis2] == 1) { // can we use a reshape instead?
    if (axis1 > axis2)
      std::swap(axis1, axis2);
    bool canReshape = true;
    for (int ax = axis1 + 1; ax < axis2 && canReshape; ax++)
      canReshape &= (shape[ax] == 1);
    if (canReshape) {
      auto newShape = shape;
      newShape.set(axis1, shape[axis2]);
      newShape.set(axis2, shape[axis1]);
      //LOG(info, "SwapAxes() did a reshape from {} to {}", shape.toString(), newShape.toString());
      return reshape(x, newShape);
    }
  }
  // TODO: This is code dup from transpose(x). Implement transpose(x) as swapAxes(x, 0, 1)
  std::vector<int> axes(shape.size());
  for (int i = 0; i < axes.size(); ++i) // @TODO: use std::iota()
    axes[i] = i;
  std::swap(axes[axis1], axes[axis2]);
  return transpose(x, axes);
}

Expr cast(Expr a, Type type) {
  if(a->value_type() == type) {
    return a;
  } else {
    return Expression<CastNodeOp>(a, type);
  }
}

Expr cross_entropy(Expr a, Expr indices) {
  return Expression<CrossEntropyNodeOp>(a, indices);
}

Expr plus(const std::vector<Expr>& nodes) {
  ABORT_IF(nodes.size() > 1, "Not implemented");
  return nodes[0];
}

Expr swish(const std::vector<Expr>& nodes) {
  ABORT_IF(nodes.size() > 1, "Not implemented");
  return swish(nodes[0]);
}

Expr gelu(const std::vector<Expr>& nodes) {
  ABORT_IF(nodes.size() > 1, "Not implemented");
  return gelu(nodes[0]);
}

Expr tanh(const std::vector<Expr>& nodes) {
  return Expression<TanhNodeOp>(nodes);
}

Expr sigmoid(const std::vector<Expr>&) {
  ABORT("Not implemented");
}

Expr relu(const std::vector<Expr>& nodes) {
  ABORT_IF(nodes.size() > 1, "Not implemented");
  return relu(nodes[0]);
}

Expr leakyrelu(const std::vector<Expr>&) {
  ABORT("Not implemented");
}

Expr prelu(const std::vector<Expr>&, float /*alpha*/) {
  ABORT("Not implemented");
}

Expr sqrt(Expr a, float eps) {
  return Expression<SqrtNodeOp>(a, eps);
}

Expr square(Expr a) {
  return Expression<SquareNodeOp>(a);
}

Expr layerNorm(Expr x,
               Expr gamma,
               Expr beta /*= nullptr*/,
               float eps /*= 1e-9*/) {

  // layerNorm accumulates in float, so small eps is fine
  std::vector<Expr> nodes = {x, gamma};
  if(beta)
    nodes.push_back(beta);
  return Expression<LayerNormalizationOp>(nodes, eps);
}

Expr highway(Expr y, Expr x, Expr t) {
  std::vector<Expr> nodes = {y, x, t};
  return Expression<HighwayNodeOp>(nodes);
}

Expr highway(const std::string prefix, Expr x) {
  // clang-format off
  size_t outDim = x->shape()[-1];
  auto graph = x->graph();
  auto g = mlp::dense()
      ("prefix", prefix + "_highway_d1")
      ("dim", outDim)
      ("activation", mlp::act::sigmoid)
      .construct(graph)->apply(x);
  auto relued = mlp::dense()
      ("prefix", prefix + "_highway_d2")
      ("dim", outDim)
      ("activation", mlp::act::ReLU)
      .construct(graph)->apply(x);
  return (g * relued) + ((1 - g) * x);
  // clang-format on
}

Expr shift(Expr a, Shape shift, float padValue) {
  return Expression<ShiftNodeOp>(a, shift, padValue);
}

#ifdef CUDA_FOUND
#ifdef CUDNN

Expr avg_pooling(Expr x,
                 int height,
                 int width,
                 int padHeight,
                 int padWidth,
                 int strideHeight,
                 int strideWidth) {
  return Expression<PoolingOp>(
      x, height, width, padHeight, padWidth, strideHeight, strideWidth, "avg");
}

Expr max_pooling(Expr x,
                 int height,
                 int width,
                 int padHeight,
                 int padWidth,
                 int strideHeight,
                 int strideWidth) {
  return Expression<PoolingOp>(
      x, height, width, padHeight, padWidth, strideHeight, strideWidth, "max");
}

Expr convert2cudnnFormat(Expr x) {
  int numWords = x->shape()[0];
  int numExamples = x->shape()[1];
  int embSize = x->shape()[2];

  std::vector<IndexType> newIndeces;
  for(int b = 0; b < numExamples; ++b) {
    for(int t = 0; t < numWords; ++t) {
      newIndeces.push_back((t * numExamples) + b);
    }
  }

  auto xRows = reshape(x, {x->shape()[0] * x->shape()[1], x->shape()[2]});

  Shape outShape({numExamples, 1, numWords, embSize});
  return reshape(rows(xRows, newIndeces), outShape);
}

Expr convertFromcudnnFormat(Expr x) {
  int batchDim = x->shape()[0];
  int sentenceDim = x->shape()[2];
  int embSize = x->shape()[3];

  auto reshapedX = reshape(x, {batchDim * sentenceDim, embSize});

  std::vector<IndexType> newIndeces;
  for(int t = 0; t < sentenceDim; ++t) {
    for(int b = 0; b < batchDim; ++b) {
      newIndeces.push_back(b * sentenceDim + t);
    }
  }

  Shape shape({batchDim, sentenceDim, embSize});
  return reshape(rows(reshapedX, newIndeces), shape);
}

Expr pooling_with_masking(Expr x, Expr mask, int width, bool isEven) {
  return Expression<PoolingWithMaskingOp>(x, mask, width, isEven);
}

#endif
#endif
}  // namespace marian
