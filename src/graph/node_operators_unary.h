#pragma once

#include "graph/backend_gpu.h"
#include "graph/node.h"
#include "kernels/sparse.h"
#include "kernels/tensor_operators.h"
#include "tensors/tensor.h"
#include "gpu/functions.h"

#ifdef CUDNN

#include <cudnn.h>

#define CUDA_CALL(x)                                  \
  do {                                                \
    if((x) != cudaSuccess) {                          \
      printf("Error at %s:%d\n", __FILE__, __LINE__); \
      return EXIT_FAILURE;                            \
    }                                                 \
  } while(0)

#define CUDNN_CALL(x)                 \
  do {                                \
    if((x) != CUDNN_STATUS_SUCCESS) { \
      printf("Error (%s) at %s:%d\n", \
             cudnnGetErrorString(x),  \
             __FILE__,                \
             __LINE__);               \
    }                                 \
  } while(0)

#endif

namespace marian {

struct UnaryNodeOp : public NaryNodeOp {
  template <typename... Args>
  UnaryNodeOp(Expr a, Args... args)
      : NaryNodeOp({a}, keywords::shape = a->shape(), args...) {}

  const std::string color() { return "yellow"; }
};

struct ScalarAddNodeOp : public UnaryNodeOp {
private:
  float scalar_{0};

public:
  template <typename... Args>
  ScalarAddNodeOp(Expr a, float scalar, Args... args)
      : UnaryNodeOp(a, args...), scalar_{scalar} {}

  NodeOps forwardOps() {
    using namespace functional;
    return {NodeOp(Element(_1 = _2 + scalar_, val_, child(0)->val()))};
  }

  NodeOps backwardOps() { return {NodeOp(Add(_1, child(0)->grad(), adj_))}; }

  const std::string type() { return "scalar_add"; }
};

struct ScalarMultNodeOp : public UnaryNodeOp {
private:
  float scalar_{0};

public:
  template <typename... Args>
  ScalarMultNodeOp(Expr a, float scalar, Args... args)
      : UnaryNodeOp(a, args...), scalar_{scalar} {}

  NodeOps forwardOps() {
    using namespace functional;
    return {NodeOp(Element(_1 = scalar_ * _2, val_, child(0)->val()))};
  }

  NodeOps backwardOps() {
    using namespace functional;
    return {NodeOp(Add(scalar_ * _1, child(0)->grad(), adj_))};
  }

  const std::string type() { return "scalar_add"; }
};

struct LogitNodeOp : public UnaryNodeOp {
  template <typename... Args>
  LogitNodeOp(Args... args) : UnaryNodeOp(args...) {}

  NodeOps forwardOps() {
    using namespace functional;
    return {NodeOp(Element(_1 = logit(_2), val_, child(0)->val()))};
  }

  NodeOps backwardOps() {
    using namespace functional;
    return {NodeOp(Add(_1 * _2 * (1.0f - _2), child(0)->grad(), adj_, val_))};
  }

  const std::string type() { return "logit"; }
};

// struct Scalar2PowNodeOp : public UnaryNodeOp {
// private:
//  float scalar_{0};
//
// public:
//  template <typename... Args>
//  Scalar2PowNodeOp(Expr a, float scalar, Args... args)
//      : UnaryNodeOp(a, args...), scalar_{scalar} {}
//
//  NodeOps forwardOps() {
//    return {NodeOp(Element(_1 = Pow(_2, scalar_), val_, child(0)->val()))};
//  }
//
//  NodeOps backwardOps() {
//    return {NodeOp(Add(scalar_ * Pow(_1, scalar_ - 1.f) * _2,
//    child(0)->grad(), child(0)->val(), adj_))};
//  }
//
//  const std::string type() { return "scalar_pow2"; }
//};
//
// struct Scalar1PowNodeOp : public UnaryNodeOp {
// private:
//  float scalar_{0};
//
// public:
//  template <typename... Args>
//  Scalar1PowNodeOp(float scalar, Expr a, Args... args)
//      : UnaryNodeOp(a, args...), scalar_{scalar} {}
//
//  NodeOps forwardOps() {
//    return {NodeOp(Element(_1 = Pow(scalar_, _2), val_, child(0)->val()))};
//  }
//
//  NodeOps backwardOps() {
//    return {NodeOp(Add(Pow(scalar_, _1) * log(scalar_) * _2, child(0)->grad(),
//    child(0)->val(), adj_))};
//  }
//
//  const std::string type() { return "scalar_pow1"; }
//};

struct TanhNodeOp : public NaryNodeOp {
  TanhNodeOp(const std::vector<Expr>& nodes)
      : NaryNodeOp(nodes, keywords::shape = newShape(nodes)) {}

  Shape newShape(const std::vector<Expr>& nodes) {
    return Shape::broadcast(nodes);
  }

  NodeOps forwardOps() {
    using namespace functional;
    switch(children_.size()) {
      case 1: return {NodeOp(Element(_1 = tanh(_2), val_, child(0)->val()))};
      case 2:
        return {NodeOp(Element(
            _1 = tanh(_2 + _3), val_, child(0)->val(), child(1)->val()))};
      case 3:
        return {NodeOp(Element(_1 = tanh(_2 + _3 + _4),
                               val_,
                               child(0)->val(),
                               child(1)->val(),
                               child(2)->val()))};
      default:
        return {
          NodeOp(Element(_1 = _2 + _3 + _4,
                         val_,
                         child(0)->val(),
                         child(1)->val(),
                         child(2)->val());
                 for(int i = 3; i < children_.size(); ++i)
                     Element(_1 = _1 + _2, val_, child(i)->val());
                 Element(_1 = tanh(_1), val_);)
        };
    }
  }

  NodeOps backwardOps() {
    using namespace functional;
    NodeOps ops;
    for(int i = 0; i < children_.size(); i++) {
      ops.push_back(
          NodeOp(Add(_1 * (1.0f - (_2 * _2)), child(i)->grad(), adj_, val_)));
    }
    return ops;
  }

  const std::string color() { return "yellow"; }

  const std::string type() { return "tanh"; }
};

/**
 * Represents a <a
 * href="https://en.wikipedia.org/wiki/Rectifier_(neural_networks)">rectified
 * linear</a> node in an expression graph.
 *
 * This node implements the activation function \f$ f(x) = \max(0, x) \f$ and
 * its derivative:
 * \f[
 *   f^\prime(x) =
 *   \begin{cases}
 *     0 & \text{if } x \leq 0 \\
 *     1 & \text{if } x > 0
 *   \end{cases}
 * \f]
 */
struct ReLUNodeOp : public UnaryNodeOp {
  template <typename... Args>
  ReLUNodeOp(Args... args) : UnaryNodeOp(args...) {}

  NodeOps forwardOps() {
    // f(x) = max(0, x)
    using namespace functional;
    return {NodeOp(Element(_1 = ReLU(_2),
                           val_,            // _1 := f(x) to be calculated
                           child(0)->val()  // _2 := x
                           ))};
  }

  NodeOps backwardOps() {
    using namespace functional;
    // dJ/dx += dJ/df * binarystep(x)
    return {NodeOp(Add(_1 * ReLUback(_2),
                       child(0)->grad(),  // dJ/dx
                       adj_,              // _1 := dJ/df
                       child(0)->val()    // _2 := f(x) = max(0, x)
                       ))};
  }

  const std::string type() { return "ReLU"; }
};

/**
 * Represents a <a
 * href="https://en.wikipedia.org/wiki/Rectifier_(neural_networks)">parametric
 * rectified linear unit</a> node in an expression graph.
 * For \f$ \alpha = 0.01 \f$ (the default value) it is equivalent to Leaky
 * ReLU.
 *
 * This node implements the activation function:
 * \f[
 *   f(x, \alpha) =
 *   \begin{cases}
 *     \alpha x & \text{if } x \leq 0 \\
 *     x        & \text{if } x > 0
 *   \end{cases}
 * \f]
 *
 * and its derivative:
 * \f[
 *   f^\prime(x, \alpha) =
 *   \begin{cases}
 *     \alpha & \text{if } x \leq 0 \\
 *     1      & \text{if } x > 0
 *   \end{cases}
 * \f]
 */
struct PReLUNodeOp : public UnaryNodeOp {
  template <typename... Args>
  PReLUNodeOp(float alpha, Args... args)
      : UnaryNodeOp(args...), alpha_(alpha) {}

  NodeOps forwardOps() {
    using namespace functional;
    return {NodeOp(Element(_1 = PReLU(_2, alpha_), val_, child(0)->val()))};
  }

  NodeOps backwardOps() {
    using namespace functional;
    return {NodeOp(Add(
        _1 * PReLUback(_2, alpha_), child(0)->grad(), adj_, child(0)->val()))};
  }

  const std::string type() { return "PReLU"; }

private:
  float alpha_{0.01};
};

/**
 * Represents a <a href="https://arxiv.org/pdf/1710.05941.pdf">swish</a> node
 * in an expression graph.
 *
 * This node implements the activation function
 * \f$ f(x) = x \cdot \sigma(x) \f$
 * and its derivative
 * \f$ f^\prime(x) = f(x) + \sigma(x)(1 - f(x)) \f$ .
 *
 */
struct SwishNodeOp : public UnaryNodeOp {
  template <typename... Args>
  SwishNodeOp(Args... args) : UnaryNodeOp(args...) {}

  NodeOps forwardOps() {
    using namespace functional;
    return {NodeOp(Element(_1 = _2 * logit(_2), val_, child(0)->val()))};
  }

  NodeOps backwardOps() {
    using namespace functional;
    // dJ/dx += dJ/df * ( f(x) + sigma(x) * (1 - f(x)) )
    return {NodeOp(Add(_1 * (_3 + logit(_2) * (1.f - _3)),
                       child(0)->grad(),  // dJ/dx
                       adj_,              // _1 := dJ/df
                       child(0)->val(),   // _2 := x
                       val_               // _3 := f(x) = x*sigma(x)
                       ))};
  }

  const std::string type() { return "swish"; }
};

struct SoftmaxNodeOp : public NaryNodeOp {
  template <typename... Args>
  SoftmaxNodeOp(Expr a, Args... args)
      : NaryNodeOp(a, args...), mask_(nullptr) {}

  template <typename... Args>
  SoftmaxNodeOp(Expr a, Expr mask, Args... args)
      : NaryNodeOp({a}, args...), mask_(mask) {}

  Expr mask_;

  NodeOps forwardOps() {
    return {
        NodeOp(Softmax(val_, child(0)->val(), mask_ ? mask_->val() : nullptr))};
  }

  virtual size_t hash() {
    if(!hash_) {
      hash_ = NaryNodeOp::hash();
      if(mask_)
        boost::hash_combine(hash_, mask_->hash());
    }
    return hash_;
  }

  virtual bool equal(Expr node) {
    if(!NaryNodeOp::equal(node))
      return false;
    Ptr<SoftmaxNodeOp> cnode = std::dynamic_pointer_cast<SoftmaxNodeOp>(node);
    if(!cnode)
      return false;
    if((bool)mask_ != (bool)cnode->mask_)
      return false;
    if(mask_ && !mask_->equal(cnode->mask_))
      return false;
    return true;
  }

  NodeOps backwardOps() {
    // For each row, the Jacobian times vector is given by:
    // J * dy = p .* (dy - avg*1)
    // where avg = p'*dy and p is the softmax output (probabilities).
    //
    // For more information, see sec. 2.5 of the following reference:
    // André F. T. Martins and Ramon Astudillo.
    // "From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label
    // Classification." ICML 2016.
    // http://jmlr.org/proceedings/papers/v48/martins16.pdf

    // val_ is already masked if there is a mask, so no need to apply here.

    return {NodeOp(SoftmaxGrad(child(0)->grad(), adj_, val_))};
  }

  const std::string type() { return "softmax"; }
};

struct LogSoftmaxNodeOp : public UnaryNodeOp {
  template <typename... Args>
  LogSoftmaxNodeOp(Args... args) : UnaryNodeOp(args...) {}

  NodeOps forwardOps() { return {NodeOp(LogSoftmax(val_, child(0)->val()))}; }

  NodeOps backwardOps() {
    // Based on the description for softmax, we have logsoftmax:
    // J * dy = dy - avg*1
    // where avg = exp(p)'*dy and p is the softmax output (probabilities).
    return {NodeOp(LogSoftmaxGrad(child(0)->grad(), adj_, val_))};
  }

  const std::string type() { return "logsoftmax"; }
};

struct SumNodeOp : public UnaryNodeOp {
  int ax_;

  template <typename... Args>
  SumNodeOp(Expr a, Args... args)
      : UnaryNodeOp(a, keywords::shape = newShape(a, args...), args...) {}

  NodeOps forwardOps() { return {NodeOp(Reduce(_1, val_, child(0)->val()))}; }

  NodeOps backwardOps() { return {NodeOp(Add(_1, child(0)->grad(), adj_))}; }

  template <class... Args>
  Shape newShape(Expr a, Args... args) {
    Shape shape = a->shape();
    ax_ = shape.axis(keywords::Get(keywords::axis, -1, args...));

    shape.set(ax_, 1);
    return shape;
  }

  const std::string type() { return "sum"; }

  const std::string color() { return "orange"; }

  virtual size_t hash() {
    if(!hash_) {
      hash_ = NaryNodeOp::hash();
      boost::hash_combine(hash_, ax_);
    }
    return hash_;
  }

  virtual bool equal(Expr node) {
    if(!NaryNodeOp::equal(node))
      return false;
    Ptr<SumNodeOp> cnode = std::dynamic_pointer_cast<SumNodeOp>(node);
    if(!cnode)
      return false;
    if(ax_ != cnode->ax_)
      return false;
    return true;
  }
};

struct MeanNodeOp : public UnaryNodeOp {
  int ax_;

  template <typename... Args>
  MeanNodeOp(Expr a, Args... args)
      : UnaryNodeOp(a, keywords::shape = newShape(a, args...), args...) {}

  NodeOps forwardOps() {
    using namespace functional;
    int left = child(0)->shape().elements() / val_->shape().elements();
    float scale = 1.f / left;

    return {NodeOp(Reduce(_1, scale, val_, child(0)->val()))};
  }

  NodeOps backwardOps() {
    using namespace functional;
    int left = child(0)->shape().elements() / val_->shape().elements();
    float scale = 1.f / left;

    return {NodeOp(Add(_1, scale, child(0)->grad(), adj_))};
  }

  template <class... Args>
  Shape newShape(Expr a, Args... args) {
    Shape shape = a->shape();
    ax_ = shape.axis(keywords::Get(keywords::axis, -1, args...));
    shape.set(ax_, 1);
    return shape;
  }

  const std::string type() { return "mean"; }

  const std::string color() { return "orange"; }

  virtual size_t hash() {
    if(!hash_) {
      hash_ = NaryNodeOp::hash();
      boost::hash_combine(hash_, ax_);
    }
    return hash_;
  }

  virtual bool equal(Expr node) {
    if(!NaryNodeOp::equal(node))
      return false;
    Ptr<MeanNodeOp> cnode = std::dynamic_pointer_cast<MeanNodeOp>(node);
    if(!cnode)
      return false;
    if(ax_ != cnode->ax_)
      return false;
    return true;
  }
};

struct LogNodeOp : public UnaryNodeOp {
  template <typename... Args>
  LogNodeOp(Args... args) : UnaryNodeOp(args...) {}

  NodeOps forwardOps() {
    using namespace functional;
    return {NodeOp(Element(_1 = log(_2), val_, child(0)->val()))};
  }

  NodeOps backwardOps() {
    using namespace functional;
    return {
        NodeOp(Add(_1 * (1.f / _2), child(0)->grad(), adj_, child(0)->val()))};
  }

  const std::string type() { return "log"; }
};

struct ExpNodeOp : public UnaryNodeOp {
  template <typename... Args>
  ExpNodeOp(Args... args) : UnaryNodeOp(args...) {}

  NodeOps forwardOps() {
    using namespace functional;
    return {NodeOp(Element(_1 = exp(_2), val_, child(0)->val()))};
  }

  NodeOps backwardOps() {
    using namespace functional;
    return {NodeOp(Add(_1 * exp(_2), child(0)->grad(), adj_, child(0)->val()))};
  }

  const std::string type() { return "exp"; }
};

struct SqrtNodeOp : public UnaryNodeOp {
  float epsilon_;

  template <typename... Args>
  SqrtNodeOp(Expr a, float epsilon, Args... args)
      : UnaryNodeOp(a, args...), epsilon_(epsilon) {}

  NodeOps forwardOps() {
    using namespace functional;
    return {NodeOp(Element(_1 = sqrt(_2 + epsilon_), val_, child(0)->val()))};
  }

  NodeOps backwardOps() {
    using namespace functional;
    return {NodeOp(Add(0.5f * (1.f / _1) * _2, child(0)->grad(), val_, adj_))};
  }

  const std::string type() { return "sqrt"; }

  virtual size_t hash() {
    if(!hash_) {
      size_t seed = NaryNodeOp::hash();
      boost::hash_combine(seed, epsilon_);
      hash_ = seed;
    }
    return hash_;
  }

  virtual bool equal(Expr node) {
    if(!NaryNodeOp::equal(node))
      return false;
    Ptr<SqrtNodeOp> cnode = std::dynamic_pointer_cast<SqrtNodeOp>(node);
    if(!cnode)
      return false;
    if(epsilon_ != cnode->epsilon_)
      return false;
    return true;
  }
};

struct SquareNodeOp : public UnaryNodeOp {
  float epsilon_;

  template <typename... Args>
  SquareNodeOp(Args... args) : UnaryNodeOp(args...) {}

  NodeOps forwardOps() {
    using namespace functional;
    return {NodeOp(Element(_1 = _2 * _2, val_, child(0)->val()))};
  }

  NodeOps backwardOps() {
    using namespace functional;
    return {
        NodeOp(Add(2.f * _1 * _2, child(0)->grad(), child(0)->val(), adj_))};
  }

  const std::string type() { return "square"; }
};

struct NegNodeOp : public UnaryNodeOp {
  template <typename... Args>
  NegNodeOp(Args... args) : UnaryNodeOp(args...) {}

  NodeOps forwardOps() {
    using namespace functional;
    return {NodeOp(Element(_1 = -_2, val_, child(0)->val()))};
  }

  NodeOps backwardOps() {
    using namespace functional;
    return {NodeOp(Add(-_1, child(0)->grad(), adj_))};
  }

  const std::string type() { return "-"; }
};

struct RowsNodeOp : public UnaryNodeOp {
  template <typename... Args>
  RowsNodeOp(Expr a, const std::vector<size_t>& indeces, Args... args)
      : UnaryNodeOp(a, keywords::shape = newShape(a, indeces), args...),
        indeces_(indeces) {}

  NodeOps forwardOps() {
    // @TODO: solve this with a tensor!

    return {NodeOp(CopyRows(val_, child(0)->val(), indeces_))};
  }

  NodeOps backwardOps() {
    return {NodeOp(PasteRows(child(0)->grad(), adj_, indeces_))};
  }

  template <class... Args>
  Shape newShape(Expr a, const std::vector<size_t>& indeces) {
    Shape shape = a->shape();
    ABORT_IF(shape.size() != 2,
             "rows operator can only be used with 2-dimensional tensors");
    shape.set(0, indeces.size());
    return shape;
  }

  const std::string type() { return "rows"; }

  const std::string color() { return "orange"; }

  virtual size_t hash() {
    if(!hash_) {
      size_t seed = NaryNodeOp::hash();
      for(auto i : indeces_)
        boost::hash_combine(seed, i);
      hash_ = seed;
    }
    return hash_;
  }

  virtual bool equal(Expr node) {
    if(!NaryNodeOp::equal(node))
      return false;
    Ptr<RowsNodeOp> cnode = std::dynamic_pointer_cast<RowsNodeOp>(node);
    if(!cnode)
      return false;
    if(indeces_ != cnode->indeces_)
      return false;
    return true;
  }

  std::vector<size_t> indeces_;
};

struct ColsNodeOp : public UnaryNodeOp {
  template <typename... Args>
  ColsNodeOp(Expr a, const std::vector<size_t>& indeces, Args... args)
      : UnaryNodeOp(a, keywords::shape = newShape(a, indeces), args...),
        indeces_(indeces) {}

  NodeOps forwardOps() {
    // @TODO: solve this with a tensor!

    return {NodeOp(CopyCols(val_, child(0)->val(), indeces_))};
  }

  NodeOps backwardOps() {
    return {NodeOp(PasteCols(child(0)->grad(), adj_, indeces_))};
  }

  template <class... Args>
  Shape newShape(Expr a, const std::vector<size_t>& indeces) {
    Shape shape = a->shape();
    shape.set(1, indeces.size());
    return shape;
  }

  const std::string type() { return "cols"; }

  const std::string color() { return "orange"; }

  virtual size_t hash() {
    if(!hash_) {
      size_t seed = NaryNodeOp::hash();
      for(auto i : indeces_)
        boost::hash_combine(seed, i);
      hash_ = seed;
    }
    return hash_;
  }

  virtual bool equal(Expr node) {
    if(!NaryNodeOp::equal(node))
      return false;
    Ptr<ColsNodeOp> cnode = std::dynamic_pointer_cast<ColsNodeOp>(node);
    if(!cnode)
      return false;
    if(indeces_ != cnode->indeces_)
      return false;
    return true;
  }

  std::vector<size_t> indeces_;
};

struct SelectNodeOp : public UnaryNodeOp {
  SelectNodeOp(Expr a, int axis, const std::vector<size_t>& indeces)
      : UnaryNodeOp(a, keywords::shape = newShape(a, axis, indeces)),
        indeces_(indeces) {}

  NodeOps forwardOps() {
    return {NodeOp(
        Select(graph()->allocator(), val_, child(0)->val(), axis_, indeces_))};
  }

  NodeOps backwardOps() {
    return {NodeOp(
        Insert(graph()->allocator(), child(0)->grad(), adj_, axis_, indeces_))};
  }

  Shape newShape(Expr a, int axis, const std::vector<size_t>& indeces) {
    Shape shape = a->shape();
    axis_ = shape.axis(axis);
    shape.set(axis_, indeces.size());
    return shape;
  }

  const std::string type() { return "select"; }

  const std::string color() { return "orange"; }

  virtual size_t hash() {
    if(!hash_) {
      size_t seed = NaryNodeOp::hash();
      boost::hash_combine(seed, axis_);
      for(auto i : indeces_)
        boost::hash_combine(seed, i);
      hash_ = seed;
    }
    return hash_;
  }

  virtual bool equal(Expr node) {
    if(!NaryNodeOp::equal(node))
      return false;
    Ptr<SelectNodeOp> cnode = std::dynamic_pointer_cast<SelectNodeOp>(node);
    if(!cnode)
      return false;
    if(axis_ != cnode->axis_)
      return false;
    if(indeces_ != cnode->indeces_)
      return false;
    return true;
  }

  std::vector<size_t> indeces_;
  int axis_{0};
};

struct TransposeNodeOp : public UnaryNodeOp {
  std::vector<int> axes_;

  TransposeNodeOp(Expr a, const std::vector<int>& axes)
      : UnaryNodeOp(a, keywords::shape = newShape(a, axes)),
        axes_{axes} {}

  NodeOps forwardOps() {
    return {NodeOp(TransposeND(val_, child(0)->val(), axes_))};
  }

  NodeOps backwardOps() {
    return {NodeOp(TransposeND(child(0)->grad(), adj_, axes_))};
  }

  template <class... Args>
  Shape newShape(Expr a, const std::vector<int>& axes) {
    Shape shape = a->shape();

    ABORT_IF(shape.size() != axes.size(),
            "Shape and transpose axes have different number of dimensions");

    for(int i = 0; i < shape.size(); ++i)
      shape.set(i, a->shape()[axes[i]]);

    return shape;
  }

  virtual size_t hash() {
    if(!hash_) {
      size_t seed = NaryNodeOp::hash();
      for(auto ax : axes_)
        boost::hash_combine(seed, ax);
      hash_ = seed;
    }
    return hash_;
  }

  virtual bool equal(Expr node) {
    if(!NaryNodeOp::equal(node))
      return false;
    Ptr<TransposeNodeOp> cnode
        = std::dynamic_pointer_cast<TransposeNodeOp>(node);
    if(!cnode)
      return false;
    if(axes_ != cnode->axes_)
      return false;
    return true;
  }

  const std::string type() { return "transpose"; }

  const std::string color() { return "orange"; }
};

class ReshapeNodeOp : public UnaryNodeOp {
private:
  Expr reshapee_;

public:
  template <typename... Args>
  ReshapeNodeOp(Expr a, Shape shape, Args... args)
      : UnaryNodeOp(a, keywords::shape = shape, args...), reshapee_(a) {
    Node::destroy_ = false;
  }

  ~ReshapeNodeOp() {}

  size_t allocate() { return 0; }
  void free() {}

  void forward() {}
  void backward() {}

  void init_dependent() { reshapee_->init_dependent(); }

  void set_zero_adjoint() { reshapee_->set_zero_adjoint(); }

  Tensor& val() {
    auto childVal = reshapee_->val();
    val_.reset(
        new TensorBase(childVal->memory(), shape(), childVal->getDevice()));
    return val_;
  };

  Tensor& grad() {
    auto childGrad = reshapee_->grad();
    adj_.reset(
        new TensorBase(childGrad->memory(), shape(), childGrad->getDevice()));
    return adj_;
  };

  const std::string type() { return "reshape"; }

  const std::string color() { return "grey"; }

  virtual size_t hash() {
    if(!hash_) {
      size_t seed = NaryNodeOp::hash();
      for(auto s : shape())
        boost::hash_combine(seed, s);
      hash_ = seed;
    }
    return hash_;
  }

  virtual bool equal(Expr node) {
    if(!NaryNodeOp::equal(node))
      return false;
    Ptr<ReshapeNodeOp> cnode = std::dynamic_pointer_cast<ReshapeNodeOp>(node);
    if(!cnode)
      return false;
    if(shape() != cnode->shape())
      return false;
    return true;
  }
};

class StepNodeOp : public UnaryNodeOp {
private:
  Expr stepNode_;
  int step_;
  int axis_;

public:
  StepNodeOp(Expr a, int step, int axis)
      : UnaryNodeOp(a, keywords::shape = newShape(a, axis)),
        stepNode_(a),
        step_(step) {
    Node::destroy_ = false;
  }

  Shape newShape(Expr a, int axis) {
    Shape outShape = a->shape();

    axis_ = outShape.axis(axis);
    for(int i = 0; i <= axis_; ++i)
      outShape.set(i, 1);

    return outShape;
  }

  size_t allocate() { return 0; }
  void free() {}

  void forward() {}
  void backward() {}

  void init_dependent() { stepNode_->init_dependent(); }

  void set_zero_adjoint() { stepNode_->set_zero_adjoint(); }

  Tensor& val() {
    auto childVal = stepNode_->val();
    size_t offset = step_ * shape().elements() * sizeof(float);
    auto mem = New<MemoryPiece>(childVal->memory()->data() + offset,
                                childVal->memory()->size());
    val_.reset(new TensorBase(mem, shape(), childVal->getDevice()));
    return val_;
  };

  Tensor& grad() {
    auto childGrad = stepNode_->grad();
    size_t offset = step_ * shape().elements() * sizeof(float);
    auto mem = New<MemoryPiece>(childGrad->memory()->data() + offset,
                                childGrad->memory()->size());
    adj_.reset(new TensorBase(mem, shape(), childGrad->getDevice()));
    return adj_;
  };

  const std::string type() { return "step"; }

  const std::string color() { return "grey"; }

  virtual size_t hash() {
    if(!hash_) {
      hash_ = NaryNodeOp::hash();
      boost::hash_combine(hash_, step_);
      boost::hash_combine(hash_, axis_);
    }
    return hash_;
  }

  virtual bool equal(Expr node) {
    if(!NaryNodeOp::equal(node))
      return false;
    Ptr<StepNodeOp> cnode = std::dynamic_pointer_cast<StepNodeOp>(node);
    if(!cnode)
      return false;
    if(step_ != cnode->step_)
      return false;
    if(axis_ != cnode->axis_)
      return false;
    return true;
  }
};

struct ShiftNodeOp : public UnaryNodeOp {
  template <typename... Args>
  ShiftNodeOp(Expr a, Shape shift, Args... args)
      : UnaryNodeOp(a, keywords::shape = a->shape(), args...), shift_(shift) {}

  NodeOps forwardOps() {
    return {NodeOp(Shift(val_, child(0)->val(), shift_))};
  }

  NodeOps backwardOps() {
    return {NodeOp(Shift(child(0)->grad(), adj_, shift_, true))};
  }

  const std::string type() { return "shift"; }

  virtual size_t hash() {
    if(!hash_) {
      size_t seed = NaryNodeOp::hash();
      for(auto i : shift_)
        boost::hash_combine(seed, i);
      hash_ = seed;
    }
    return hash_;
  }

  virtual bool equal(Expr node) {
    if(!NaryNodeOp::equal(node))
      return false;
    Ptr<ShiftNodeOp> cnode = std::dynamic_pointer_cast<ShiftNodeOp>(node);
    if(!cnode)
      return false;
    if(shift_ != cnode->shift_)
      return false;
    return true;
  }

  Shape shift_;
};

// struct LexicalProbNodeOp : public NaryNodeOp {
//  template <typename... Args>
//  LexicalProbNodeOp(
//      Expr logits, Expr att, float eps, Ptr<sparse::CSR> lf, Args... args)
//      : NaryNodeOp({logits, att}, keywords::shape = logits->shape(), args...),
//        eps_(eps),
//        lf_(lf) {}
//
//  void forward() {
//    sparse::LfaForward(val_, child(0)->val(), child(1)->val(), lf_);
//    // val = x + ln(p + eps)
//    Element(_1 = (log(_1 + eps_) + _2), val_, child(0)->val());
//  }
//
//  void backward() {
//    Add(_1, child(0)->grad(), adj_);
//    // adj' = adj / (p + eps) = adj / exp(val - x)
//    Element(_1 = _1 / exp(_2 - _3), adj_, val_, child(0)->val());
//    sparse::LfaBackward(child(1)->grad(), adj_, lf_);
//  }
//
//  const std::string type() { return "lexical_prob"; }
//
//  virtual size_t hash() {
//    if(!hash_) {
//      size_t seed = NaryNodeOp::hash();
//      boost::hash_combine(seed, (size_t)lf_.get());
//      hash_ = seed;
//    }
//    return hash_;
//  }
//
//  float eps_;
//  Ptr<sparse::CSR> lf_;
//};

#ifdef CUDNN

class PoolingOp : public UnaryNodeOp {
public:
  enum class Mode { MAX_POOLING, AVERAGE_POOLING };

  PoolingOp(Expr x,
            int height,
            int width,
            int padHeight,
            int padWidth,
            int strideHeight,
            int strideWidth,
            Mode mode = Mode::AVERAGE_POOLING)
      : UnaryNodeOp(x) {
    CUDNN_CALL(cudnnCreate(&cudnnHandle_));

    CUDNN_CALL(cudnnCreateTensorDescriptor(&xDesc_));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(xDesc_,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT,
                                          x->shape()[0],
                                          x->shape()[1],
                                          x->shape()[2],
                                          x->shape()[3]));

    cudnnPoolingMode_t cudnnPoolingMode;
    switch(mode) {
      case Mode::MAX_POOLING: cudnnPoolingMode = CUDNN_POOLING_MAX; break;
      case Mode::AVERAGE_POOLING:
        cudnnPoolingMode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
        break;
      default: break;
    };

    height = std::min(height, x->shape()[2]);
    strideHeight = std::min(strideHeight, x->shape()[2]);

    CUDNN_CALL(cudnnCreatePoolingDescriptor(&poolingDesc_));
    CUDNN_CALL(cudnnSetPooling2dDescriptor(poolingDesc_,
                                           cudnnPoolingMode,
                                           CUDNN_NOT_PROPAGATE_NAN,
                                           height,
                                           width,
                                           padHeight,
                                           padWidth,
                                           strideHeight,
                                           strideWidth));
   /* @TODO: does not compile
    CUDNN_CALL(cudnnGetPooling2dForwardOutputDim(poolingDesc_,
                                                 xDesc_,
                                                 shape_.begin(),
                                                 shape_.begin() + 1,
                                                 shape_.begin() + 2,
                                                 shape_.begin() + 3));
*/
    CUDNN_CALL(cudnnCreateTensorDescriptor(&yDesc_));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(yDesc_,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT,
                                          shape_[0],
                                          shape_[1],
                                          shape_[2],
                                          shape_[3]));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&adjDesc_));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(adjDesc_,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT,
                                          shape_[0],
                                          shape_[1],
                                          shape_[2],
                                          shape_[3]));
  }

  NodeOps forwardOps() {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cudaSetDevice(val_->getDevice());

    return {NodeOp(CUDNN_CALL(cudnnPoolingForward(cudnnHandle_,
                                                  poolingDesc_,
                                                  &alpha,
                                                  xDesc_,
                                                  children_[0]->val()->data(),
                                                  &beta,
                                                  yDesc_,
                                                  val_->data())))};
  }

  NodeOps backwardOps() {
    cudaSetDevice(adj_->getDevice());
    const float alpha = 1.0f;
    const float beta = 1.0f;
    return {
        NodeOp(CUDNN_CALL(cudnnPoolingBackward(cudnnHandle_,
                                               poolingDesc_,
                                               &alpha,
                                               yDesc_,
                                               val_->data(),
                                               adjDesc_,
                                               adj_->data(),
                                               xDesc_,
                                               children_[0]->val()->data(),
                                               &beta,
                                               xDesc_,
                                               children_[0]->grad()->data())))};
  }

  const std::string type() { return "layer_max_pooling"; }

  virtual ~PoolingOp() {
    CUDNN_CALL(cudnnDestroy(cudnnHandle_));
    CUDNN_CALL(cudnnDestroyPoolingDescriptor(poolingDesc_));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(xDesc_));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(yDesc_));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(adjDesc_));
  }

protected:
  cudnnHandle_t cudnnHandle_;
  cudnnPoolingDescriptor_t poolingDesc_;
  cudnnTensorDescriptor_t xDesc_;
  cudnnTensorDescriptor_t yDesc_;
  cudnnTensorDescriptor_t adjDesc_;
};

#endif
}
