#pragma once

#include "graph/node.h"
#include "tensors/tensor.h"
#include "kernels/tensor_operators.h"
#include "kernels/thrust_functions.h"

namespace marian {

struct UnaryNodeOp : public NaryNodeOp {
    template <typename ...Args>
    UnaryNodeOp(Expr a, Args ...args)
    : NaryNodeOp({a},
                keywords::shape=a->shape(),
                args...) {}

    const std::string color() {
      return "yellow";
    }
};

struct LogitNodeOp : public UnaryNodeOp {
  template <typename ...Args>
  LogitNodeOp(Args ...args)
  : UnaryNodeOp(args...) {  }

  NodeOps forwardOps() {
    return {
      NodeOp(Element(_1 = Sigma(_2),
                     val_,
                     children_[0]->val()))
    };
  }

  NodeOps backwardOps() {
    return {
      NodeOp(Add(_1 * _2 * (1.0f - _2),
                 children_[0]->grad(),
                 adj_, val_))
    };
  }

  const std::string type() {
    return "logit";
  }
};

struct TanhNodeOp : public NaryNodeOp {
  TanhNodeOp(const std::vector<Expr>& nodes)
    : NaryNodeOp(nodes, keywords::shape=newShape(nodes)) { }

  Shape newShape(const std::vector<Expr>& nodes) {
    Shape shape = nodes[0]->shape();

    for(int n = 1; n < nodes.size(); ++n) {
      Shape shapen = nodes[n]->shape();
      for(int i = 0; i < shapen.size(); ++i) {
        UTIL_THROW_IF2(shape[i] != shapen[i] && shape[i] != 1 && shapen[i] != 1,
                       "Shapes cannot be broadcasted");
        shape.set(i, std::max(shape[i], shapen[i]));
      }
    }
    return shape;
  }

  NodeOps forwardOps() {
    switch (children_.size()) {
      case 1:
        return { NodeOp(Element(_1 = Tanh(_2),
                                val_,
                                children_[0]->val())) };
      case 2:
        return { NodeOp(Element(_1 = Tanh(_2 + _3),
                                val_,
                                children_[0]->val(),
                                children_[1]->val())) };
      case 3:
        return { NodeOp(Element(_1 = Tanh(_2 + _3 + _4),
                                val_,
                                children_[0]->val(),
                                children_[1]->val(),
                                children_[2]->val())) };
      default:
        return {
          NodeOp(
            Element(_1 = _2 + _3 + _4,
                    val_,
                    children_[0]->val(),
                    children_[1]->val(),
                    children_[2]->val());
            for(int i = 3; i < children_.size(); ++i)
              Element(_1 += _2, val_, children_[i]->val());
            Element(_1 = Tanh(_1), val_);
          )
        };
    }
  }

  NodeOps backwardOps() {
    NodeOps ops;
    for(auto&& child : children_) {
      ops.push_back(
        NodeOp(Add(_1 * (1.0f - (_2 * _2)),
                   child->grad(), adj_, val_))
      );
    }
    return ops;
  }

  const std::string color() {
    return "yellow";
  }

  const std::string type() {
    return "tanh";
  }
};

/**
 * Represents a <a href="https://en.wikipedia.org/wiki/Rectifier_(neural_networks)">rectified linear</a> node
 *        in an expression graph.
 *
 * This node implements the <a href="https://en.wikipedia.org/wiki/Activation_function">activation function</a>
 *        \f$f(x) = \max(0, x)\f$ and its derivative:
 *
 \f[
 f^\prime(x) =
  \begin{cases}
   0 & \text{if } x \leq 0 \\
   1 & \text{if } x > 0
  \end{cases}
\f]
 */
struct ReLUNodeOp : public UnaryNodeOp {
  template <typename ...Args>
  ReLUNodeOp(Args ...args)
  : UnaryNodeOp(args...) { }

  NodeOps forwardOps() {
    return {
      NodeOp(Element(_1 = ReLU(_2),
                     val_,
                     children_[0]->val()))
    };
  }

  NodeOps backwardOps() {
    return {
      NodeOp(Add(_1 * ReLUback(_2),
                 children_[0]->grad(),
                 adj_, children_[0]->val()))
    };
  }

  const std::string type() {
    return "ReLU";
  }
};

/**
 * @brief Represents a <a href="https://en.wikipedia.org/wiki/Dropout_(neural_networks)">dropout</a> node
 *        in an expression graph.
 *
 * @see \cite dropout
 * @see \cite cudnn
 */
//struct DropoutNodeOp : public UnaryNodeOp {
//  template <typename ...Args>
//  DropoutNodeOp(Args ...args)
//  : UnaryNodeOp(args...),
//    allocated_(false), p_(Get(keywords::value, 0.5)) {}
//
//  ~DropoutNodeOp() {
//    if(allocated_)
//      CudnnDropoutDestroy(dropDesc_, space_, states_);
// }
//
//  void inference() {
//    Element(_1 = _2, val_, children_[0]->val());
//  }
//
//  void forward() {
//    if(!allocated_) {
//        CudnnDropoutPrepare(children_[0]->val(), p_,
//                            &dropDesc_,
//                            &space_, &spaceSize_,
//                            &states_, (size_t)this); // seeding with pointer address
//        allocated_ = true;
//    }
//
//    CudnnDropoutForward(dropDesc_, space_, spaceSize_,
//                        val_, children_[0]->val());
//  }
//
//  void backward() {
//    if(children_[0]->trainable())
//        CudnnDropoutBackward(dropDesc_, space_, spaceSize_,
//                             children_[0]->grad(), adj_);
//  }
//
//  const std::string type() {
//    return "dropout";
//  }
//
//  private:
//    bool allocated_;
//    float p_;
//    void* states_;
//    void* space_;
//    size_t spaceSize_;
//    cudnnDropoutDescriptor_t dropDesc_;
//};

struct SoftmaxNodeOp : public NaryNodeOp {
  template <typename ...Args>
  SoftmaxNodeOp(Expr a, Args ...args)
    : NaryNodeOp(a, args...), mask_(nullptr) {
  }

  template <typename ...Args>
  SoftmaxNodeOp(Expr a, Expr mask, Args ...args)
    : NaryNodeOp({a, mask}, args...), mask_(mask) {
  }

  Expr mask_;

  NodeOps forwardOps() {
    return {
      NodeOp(Softmax(val_,
                     children_[0]->val(),
                     mask_ ? mask_->val() : nullptr))
    };
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

    return {
      NodeOp(SoftmaxGrad(children_[0]->grad(), adj_, val_))
    };
  }

  const std::string type() {
    return "softmax";
  }
};

struct LogSoftmaxNodeOp : public UnaryNodeOp {
  template <typename ...Args>
    LogSoftmaxNodeOp(Args ...args)
    : UnaryNodeOp(args...) { }

  NodeOps forwardOps() {
    return {
      NodeOp(LogSoftmax(val_, children_[0]->val()))
    };
  }

  NodeOps backwardOps() {
    // Based on the description for softmax, we have logsoftmax:
    // J * dy = dy - avg*1
    // where avg = exp(p)'*dy and p is the softmax output (probabilities).
    return {
      NodeOp(LogSoftmaxGrad(children_[0]->grad(), adj_, val_))
    };
  }

  const std::string type() {
    return "logsoftmax";
  }
};

struct SumNodeOp : public UnaryNodeOp {
  template <typename ...Args>
  SumNodeOp(Expr a, Args ...args)
    : UnaryNodeOp(a, keywords::shape=newShape(a, args...), args...) { }

  NodeOps forwardOps() {
    return { NodeOp(Reduce(_1, val_, children_[0]->val())) };
  }

  NodeOps backwardOps() {
    return { NodeOp(Add(_1, children_[0]->grad(), adj_)) };
  }

  template <class ...Args>
  Shape newShape(Expr a, Args ...args) {
    int ax = keywords::Get(keywords::axis, -1, args...);
    Shape shape = a->shape();
    if(ax != -1) {
      shape.set(ax, 1);
    }
    else {
      shape.set(0, 1);
      shape.set(1, 1);
      shape.set(2, 1);
      shape.set(3, 1);
    }
    return shape;
  }

  const std::string type() {
    return "sum";
  }

  const std::string color() {
    return "orange";
  }

};

struct MeanNodeOp : public UnaryNodeOp {
  template <typename ...Args>
  MeanNodeOp(Expr a, Args ...args)
    : UnaryNodeOp(a, keywords::shape=newShape(a, args...), args...) { }

  NodeOps forwardOps() {
    int left = children_[0]->shape().elements() / val_->shape().elements();
    float scale = 1.f / left;

    return {
      NodeOp(Reduce(_1 * scale, val_, children_[0]->val()))
    };
  }

  NodeOps backwardOps() {
    int left = children_[0]->shape().elements() / val_->shape().elements();
    float scale = 1.f / left;

    return {
      NodeOp(Add(_1 * scale, children_[0]->grad(), adj_))
    };
  }

  template <class ...Args>
  Shape newShape(Expr a, Args ...args) {
    int ax = keywords::Get(keywords::axis, -1, args...);
    Shape shape = a->shape();
    if(ax != -1) {
      shape.set(ax, 1);
    }
    else {
      shape.set(0, 1);
      shape.set(1, 1);
      shape.set(2, 1);
      shape.set(3, 1);
    }
    return shape;
  }

  const std::string type() {
    return "mean";
  }

  const std::string color() {
    return "orange";
  }
};


struct LogNodeOp : public UnaryNodeOp {
  template <typename ...Args>
  LogNodeOp(Args ...args)
  : UnaryNodeOp(args...) {}

  NodeOps forwardOps() {
    return {
      NodeOp(Element(_1 = Log(_2),
                     val_,
                     children_[0]->val()))
    };
  }

  NodeOps backwardOps() {
    return {
      NodeOp(Add(_1 * (1.f / _2),
                 children_[0]->grad(),
                 adj_,
                 children_[0]->val()))
    };
  }

  const std::string type() {
    return "log";
  }
};

struct ExpNodeOp : public UnaryNodeOp {
  template <typename ...Args>
    ExpNodeOp(Args ...args)
    : UnaryNodeOp(args...) { }

  NodeOps forwardOps() {
    return {
      NodeOp(Element(_1 = Exp(_2),
                     val_,
                     children_[0]->val()))
    };
  }

  NodeOps backwardOps() {
    return {
      NodeOp(Add(_1 * Exp(_2),
                 children_[0]->grad(),
                 adj_,
                 children_[0]->val()))
    };
  }

  const std::string type() {
    return "exp";
  }

};

struct SqrtNodeOp : public UnaryNodeOp {
  float epsilon_;

  template <typename ...Args>
    SqrtNodeOp(Expr a, float epsilon, Args ...args)
    : UnaryNodeOp(a, args...),
      epsilon_(epsilon) { }

  NodeOps forwardOps() {
    return {
      NodeOp(Element(_1 = Sqrt(_2 + epsilon_),
                     val_,
                     children_[0]->val()))
    };
  }

  NodeOps backwardOps() {
    return {
      NodeOp(Add(0.5f * (1.f / _1) * _2,
                 children_[0]->grad(),
                 val_,
                 adj_))
    };
  }

  const std::string type() {
    return "sqrt";
  }

};

struct SquareNodeOp : public UnaryNodeOp {
  float epsilon_;

  template <typename ...Args>
    SquareNodeOp(Args ...args)
    : UnaryNodeOp(args...) { }

  NodeOps forwardOps() {
    return {
      NodeOp(Element(_1 = _2 * _2,
                     val_,
                     children_[0]->val()))
    };
  }

  NodeOps backwardOps() {
    return {
      NodeOp(Add(2.f * _1 * _2,
                 children_[0]->grad(),
                 children_[0]->val(),
                 adj_))
    };
  }

  const std::string type() {
    return "square";
  }

};


struct NegNodeOp : public UnaryNodeOp {
  template <typename ...Args>
  NegNodeOp(Args ...args)
  : UnaryNodeOp(args...) { }

  NodeOps forwardOps() {
    return {
      NodeOp(Element(_1 = -_2,
                     val_,
                     children_[0]->val()))
    };
  }

  NodeOps backwardOps() {
    return {
      NodeOp(Add(-_1,
                 children_[0]->grad(),
                 adj_))
    };
  }

  const std::string type() {
    return "-";
  }
};

struct RowsNodeOp : public UnaryNodeOp {
  template <typename ...Args>
  RowsNodeOp(Expr a, const std::vector<size_t>& indeces, Args ...args)
    : UnaryNodeOp(a, keywords::shape=newShape(a, indeces), args...),
      indeces_(indeces) {
  }

  NodeOps forwardOps() {
    // @TODO: solve this with a tensor!

    return {
      NodeOp(CopyRows(val_,
                      children_[0]->val(),
                      indeces_))
    };
  }

  NodeOps backwardOps() {
    return {
      NodeOp(PasteRows(children_[0]->grad(),
                       adj_,
                       indeces_))
    };
  }

  template <class ...Args>
  Shape newShape(Expr a, const std::vector<size_t>& indeces) {
    Shape shape = a->shape();
    shape.set(0, indeces.size());
    return shape;
  }

  const std::string type() {
    return "rows";
  }

  const std::string color() {
    return "orange";
  }

  std::vector<size_t> indeces_;
};

struct TransposeNodeOp : public UnaryNodeOp {
  template <typename ...Args>
  TransposeNodeOp(Expr a, Args ...args)
    : UnaryNodeOp(a, keywords::shape=newShape(a), args...) { }

  NodeOps forwardOps() {
    return {
      NodeOp(Transpose(getCublasHandle(),
                       val_, children_[0]->val()))
    };
  }

  NodeOps backwardOps() {
    return {
      NodeOp(Transpose(getCublasHandle(),
                       children_[0]->grad(), adj_))
    };
  }

  template <class ...Args>
  Shape newShape(Expr a) {
    Shape shape = a->shape();
    int temp = shape[0];
    shape.set(0, shape[1]);
    shape.set(1, temp);
    return shape;
  }

  const std::string type() {
    return "transpose";
  }

  const std::string color() {
    return "orange";
  }
};

struct ReshapeNodeOp : public UnaryNodeOp {
  template <typename ...Args>
  ReshapeNodeOp(Expr a, Shape shape, Args ...args)
    : UnaryNodeOp(a, keywords::shape=shape, args...) { }

  size_t allocate() { return 0; }
  void free() {}

  void forward() {}
  void backward() {}

  void init_dependent() {
    children_[0]->init_dependent();
  }

  void set_zero_adjoint() {
    children_[0]->set_zero_adjoint();
  }

  Tensor& val()  {
    auto childVal = children_[0]->val();
    val_.reset(new TensorBase(childVal->data(), shape(), childVal->getDevice()));
    return val_;
  };

  Tensor& grad() {
    auto childGrad = children_[0]->grad();
    adj_.reset(new TensorBase(childGrad->data(), shape(), childGrad->getDevice()));
    return adj_;
  };

  const std::string type() {
    return "reshape";
  }

  const std::string color() {
    return "grey";
  }
};

struct TimestepNodeOp : public UnaryNodeOp {
  size_t step_;

  TimestepNodeOp(Expr a, size_t step)
    : UnaryNodeOp(a, keywords::shape=newShape(a)),
      step_(step)
    { }

  Shape newShape(Expr a) {
    Shape outShape = a->shape();
    outShape.set(2, 1);
    outShape.set(3, 1);
    return outShape;
  }

  size_t allocate() { return 0; }
  void free() {}

  void forward() {}
  void backward() {}

  void init_dependent() {
    children_[0]->init_dependent();
  }

  void set_zero_adjoint() {
    children_[0]->set_zero_adjoint();
  }

  Tensor& val()  {
    auto childVal = children_[0]->val();
    size_t offset = step_ * shape().elements();
    val_.reset(new TensorBase(childVal->data() + offset, shape(), childVal->getDevice()));
    return val_;
  };

  Tensor& grad() {
    auto childGrad = children_[0]->grad();
    size_t offset = step_ * shape().elements();
    adj_.reset(new TensorBase(childGrad->data() + offset, shape(), childGrad->getDevice()));
    return adj_;
  };

  const std::string type() {
    return "step";
  }

  const std::string color() {
    return "grey";
  }
};

}
