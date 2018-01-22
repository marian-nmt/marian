#include "graph/expression_operators.h"
#include "kernels/sparse.h"
#include "layers/constructors.h"

#include "graph/node_operators.h"
#include "graph/node_operators_binary.h"
#include "graph/node_operators_unary.h"

namespace marian {

Expr debug(Expr a, const std::string& message) {
  a->debug(message);
  return a;
}

Expr logit(Expr a) {
  return Expression<LogitNodeOp>(a);
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

Expr log(Expr a) {
  return Expression<LogNodeOp>(a);
};

Expr exp(Expr a) {
  return Expression<ExpNodeOp>(a);
};

Expr swish(Expr a) {
  return Expression<SwishNodeOp>(a);
}

Expr operator-(Expr a) {
  return Expression<NegNodeOp>(a);
};

Expr softmax(Expr a, Expr mask) {
  return Expression<SoftmaxNodeOp>(a, mask);
}

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

/*********************************************************/

Expr operator+(Expr a, float b) {
  return Expression<ScalarAddNodeOp>(a, b);
}

Expr operator+(float a, Expr b) {
  return Expression<ScalarAddNodeOp>(b, a);
}

Expr operator-(Expr a, float b) {
  return Expression<ScalarAddNodeOp>(a, -b);
}

Expr operator-(float a, Expr b) {
  return Expression<ScalarAddNodeOp>(-b, a);
}

Expr operator*(float a, Expr b) {
  return Expression<ScalarMultNodeOp>(b, a);
}

Expr operator*(Expr a, float b) {
  return Expression<ScalarMultNodeOp>(a, b);
}

Expr operator/(Expr a, float b) {
  return Expression<ScalarMultNodeOp>(a, 1.f / b);
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

Expr concatenate(const std::vector<Expr>& concats, keywords::axis_k ax) {
  return Expression<ConcatenateNodeOp>(concats, ax);
}

Expr repeat(Expr a, size_t repeats, keywords::axis_k ax) {
  if(repeats == 1)
    return a;
  return concatenate(std::vector<Expr>(repeats, a), ax);
}


Expr reshape(Expr a, Shape shape) {
  return Expression<ReshapeNodeOp>(a, shape);
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
  for(int i = 1; i <= a->shape().size(); ++i)
    nShape.set(-i, a->shape()[-i]);

  return reshape(a, nShape);
}

Expr flatten(Expr a) {
  Shape shape = {a->shape().elements()};
  return Expression<ReshapeNodeOp>(a, shape);
}

Expr flatten_2d(Expr a) {
  Shape shape = {
    a->shape().elements() / a->shape()[-1],
    a->shape()[-1]
  };

  return Expression<ReshapeNodeOp>(a, shape);
}

Expr rows(Expr a, const std::vector<size_t>& indices) {
  return Expression<RowsNodeOp>(a, indices);
}

Expr cols(Expr a, const std::vector<size_t>& indices) {
  return Expression<ColsNodeOp>(a, indices);
}

Expr select(Expr a, int axis, const std::vector<size_t>& indices) {
  return Expression<SelectNodeOp>(a, axis, indices);
}

Expr sum(Expr a, keywords::axis_k ax) {
  return Expression<SumNodeOp>(a, ax);
}

Expr mean(Expr a, keywords::axis_k ax) {
  return Expression<MeanNodeOp>(a, ax);
}

Expr scalar_product(Expr a, Expr b, keywords::axis_k ax) {
  return Expression<ScalarProductNodeOp>(a, b, ax);
}

Expr weighted_average(Expr in, Expr weights, keywords::axis_k ax) {
  auto p = scalar_product(in, weights, ax);
  auto s = sum(weights, ax);
  return p / s;
}

Expr dot(Expr a, Expr b, bool transA, bool transB, float scalar) {
  return Expression<DotNodeOp>(a, b, transA, transB, scalar);
}

Expr bdot(Expr a, Expr b, bool transA, bool transB, float scalar) {
  return Expression<DotBatchedNodeOp>(a, b, transA, transB, scalar);
}

Expr transpose(Expr a) {
  std::vector<int> axes(a->shape().size());
  for(int i = 0; i < axes.size(); ++i) {
    axes[i] = i;
  }
  if(axes.size() > 1) {
    axes[axes.size() - 1] = axes.size() - 2;
    axes[axes.size() - 2] = axes.size() - 1;
  }
  return Expression<TransposeNodeOp>(a, axes);
}

Expr transpose(Expr a, const std::vector<int>& axes) {
  return Expression<TransposeNodeOp>(a, axes);
}

Expr step(Expr a, int step, int axis) {
  return Expression<StepNodeOp>(a, step, axis);
}

Expr cross_entropy(Expr a, Expr b) {
  //auto sOrig = a->shape();
  //auto sOut = a->shape();
  //Shape sTemp({sOrig[0] * sOrig[2] * sOrig[3], sOrig[1], 1, 1});
  //sOut.set(1, 1);
  //return reshape(Expression<CrossEntropyNodeOp>(reshape(a, sTemp), b), sOut);

  return Expression<CrossEntropyNodeOp>(a, b);
}

Expr affine(Expr a, Expr b, Expr c,
            bool transA, bool transB, float scalar) {
  std::vector<Expr> nodes = {a, b, c};
  return Expression<AffineNodeOp>(nodes, transA, transB, scalar);
}

Expr plus(const std::vector<Expr>&) {
  ABORT("Not implemented");
}

Expr swish(const std::vector<Expr>&) {
  ABORT("Not implemented");
}

Expr tanh(const std::vector<Expr>& nodes) {
  return Expression<TanhNodeOp>(nodes);
}

Expr logit(const std::vector<Expr>&) {
  ABORT("Not implemented");
}

Expr relu(const std::vector<Expr>&) {
  ABORT("Not implemented");
}

Expr leakyrelu(const std::vector<Expr>&) {
  ABORT("Not implemented");
}

Expr prelu(const std::vector<Expr>&, float alpha) {
  ABORT("Not implemented");
}

Expr sqrt(Expr a, float eps) {
  return Expression<SqrtNodeOp>(a, eps);
}

Expr square(Expr a) {
  return Expression<SquareNodeOp>(a);
}

Expr layer_norm(Expr x,
                Expr gamma,
                Expr beta /*= nullptr*/,
                float eps /*= 1e-9*/) {
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
  size_t outDim = x->shape()[-1];
  auto g = mlp::dense(x->graph())
      ("prefix", prefix + "_highway_d1")
      ("dim", outDim)
      ("activation", mlp::act::logit)
      .construct()->apply(x);
  auto relued = mlp::dense(x->graph())
      ("prefix", prefix + "_highway_d2")
      ("dim", outDim)
      ("activation", mlp::act::ReLU)
      .construct()->apply(x);
  return (g * relued) + ((1 - g) * x);
}

// Expr batch_norm(Expr x, Expr gamma, Expr beta) {
//  auto mju = mean(x, keywords::axis=0);
//  auto xmmju = x - mju;
//  auto std = sqrt(mean(square(xmmju), keywords::axis=0), 1e-9);
//
//  if(beta)
//    return gamma * (xmmju / std) + beta;
//  else
//    return gamma * (xmmju / std);
//}

Expr shift(Expr a, Shape shift) {
  return Expression<ShiftNodeOp>(a, shift);
}

// Expr lexical_bias(Expr logits, Expr att, float eps, Ptr<sparse::CSR> lf) {
//  return Expression<LexicalProbNodeOp>(logits, att, eps, lf);
//}

Expr avg_pooling(
    Expr x,
    int height,
    int width,
    int padHeight,
    int padWidth,
    int strideHeight,
    int strideWidth) {
  return Expression<PoolingOp>(x,
      height,
      width,
      padHeight,
      padWidth,
      strideHeight,
      strideWidth,
      "avg");
}

Expr max_pooling(
    Expr x,
    int height,
    int width,
    int padHeight,
    int padWidth,
    int strideHeight,
    int strideWidth)
{
  return Expression<PoolingOp>(x,
      height,
      width,
      padHeight,
      padWidth,
      strideHeight,
      strideWidth,
      "max");
}

Expr convert2cudnnFormat(Expr x) {
  int numWords = x->shape()[0];
  int numExamples = x->shape()[1];
  int embSize = x->shape()[2];

  std::vector<size_t> newIndeces;
  for (int b = 0; b < numExamples; ++b) {
    for (int t = 0; t < numWords; ++t) {
      newIndeces.push_back((t * numExamples) + b);
    }
  }

  auto xRows = reshape(x, {x->shape()[0] * x ->shape()[1], x->shape()[2]});

  Shape outShape({numExamples, 1, numWords, embSize});
  return reshape(rows(xRows, newIndeces), outShape);
}

Expr convertFromcudnnFormat(Expr x) {
  int batchDim = x->shape()[0];
  int sentenceDim = x->shape()[2];
  int embSize = x->shape()[3];

  auto reshapedX = reshape(x, {batchDim * sentenceDim, embSize});

  std::vector<size_t> newIndeces;
  for (int t = 0; t < sentenceDim; ++t) {
    for (int b = 0; b < batchDim; ++b) {
      newIndeces.push_back(b * sentenceDim + t);
    }
  }

  Shape shape({batchDim, sentenceDim, embSize});
  return reshape(rows(reshapedX, newIndeces), shape);
}

Expr pooling_with_masking(Expr x, Expr mask, int width, bool isEven) {
  return Expression<PoolingWithMaskingOp>(x, mask, width, isEven);
}

}
