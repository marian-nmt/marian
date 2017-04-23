#pragma once

#include "graph/node.h"
#include "tensors/tensor.h"
#include "kernels/tensor_operators.h"
#include "kernels/thrust_functions.h"
#include "kernels/sparse.h"

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
                     child(0)->val()))
    };
  }

  NodeOps backwardOps() {
    return {
      NodeOp(Add(_1 * _2 * (1.0f - _2),
                 child(0)->grad(),
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
                                child(0)->val())) };
      case 2:
        return { NodeOp(Element(_1 = Tanh(_2 + _3),
                                val_,
                                child(0)->val(),
                                child(1)->val())) };
      case 3:
        return { NodeOp(Element(_1 = Tanh(_2 + _3 + _4),
                                val_,
                                child(0)->val(),
                                child(1)->val(),
                                child(2)->val())) };
      default:
        return {
          NodeOp(
            Element(_1 = _2 + _3 + _4,
                    val_,
                    child(0)->val(),
                    child(1)->val(),
                    child(2)->val());
            for(int i = 3; i < children_.size(); ++i)
              Element(_1 += _2, val_, child(i)->val());
            Element(_1 = Tanh(_1), val_);
          )
        };
    }
  }

  NodeOps backwardOps() {
    NodeOps ops;
    for(int i = 0; i < children_.size(); i++) {
      ops.push_back(
        NodeOp(Add(_1 * (1.0f - (_2 * _2)),
                   child(i)->grad(), adj_, val_))
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
                     child(0)->val()))
    };
  }

  NodeOps backwardOps() {
    return {
      NodeOp(Add(_1 * ReLUback(_2),
                 child(0)->grad(),
                 adj_, child(0)->val()))
    };
  }

  const std::string type() {
    return "ReLU";
  }
};

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
                     child(0)->val(),
                     mask_ ? mask_->val() : nullptr))
    };
  }

  virtual size_t hash() {
    if(!hash_) {
      hash_ = NaryNodeOp::hash();
      if(mask_)
        boost::hash_combine(hash_, mask_->hash());
    }
    return hash_;
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
      NodeOp(SoftmaxGrad(child(0)->grad(), adj_, val_))
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
      NodeOp(LogSoftmax(val_, child(0)->val()))
    };
  }

  NodeOps backwardOps() {
    // Based on the description for softmax, we have logsoftmax:
    // J * dy = dy - avg*1
    // where avg = exp(p)'*dy and p is the softmax output (probabilities).
    return {
      NodeOp(LogSoftmaxGrad(child(0)->grad(), adj_, val_))
    };
  }

  const std::string type() {
    return "logsoftmax";
  }
};

struct SumNodeOp : public UnaryNodeOp {
  int ax_;

  template <typename ...Args>
  SumNodeOp(Expr a, Args ...args)
    : UnaryNodeOp(a, keywords::shape=newShape(a, args...), args...),
      ax_(keywords::Get(keywords::axis, -1, args...)) { }

  NodeOps forwardOps() {
    return { NodeOp(Reduce(_1, val_, child(0)->val())) };
  }

  NodeOps backwardOps() {
    return { NodeOp(Add(_1, child(0)->grad(), adj_)) };
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

  virtual size_t hash() {
    if(!hash_) {
      hash_ = NaryNodeOp::hash();
      boost::hash_combine(hash_, ax_);
    }
    return hash_;
  }


};

struct MeanNodeOp : public UnaryNodeOp {
  int ax_;

  template <typename ...Args>
  MeanNodeOp(Expr a, Args ...args)
    : UnaryNodeOp(a, keywords::shape=newShape(a, args...), args...),
      ax_(keywords::Get(keywords::axis, -1, args...)) { }

  NodeOps forwardOps() {
    int left = child(0)->shape().elements() / val_->shape().elements();
    float scale = 1.f / left;

    return {
      NodeOp(Reduce(_1, val_, child(0)->val(), scale))
    };
  }

  NodeOps backwardOps() {
    int left = child(0)->shape().elements() / val_->shape().elements();
    float scale = 1.f / left;

    return {
      NodeOp(Add(_1, child(0)->grad(), adj_, scale))
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

  virtual size_t hash() {
    if(!hash_) {
      hash_ = NaryNodeOp::hash();
      boost::hash_combine(hash_, ax_);
    }
    return hash_;
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
                     child(0)->val()))
    };
  }

  NodeOps backwardOps() {
    return {
      NodeOp(Add(_1 * (1.f / _2),
                 child(0)->grad(),
                 adj_,
                 child(0)->val()))
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
                     child(0)->val()))
    };
  }

  NodeOps backwardOps() {
    return {
      NodeOp(Add(_1 * Exp(_2),
                 child(0)->grad(),
                 adj_,
                 child(0)->val()))
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
                     child(0)->val()))
    };
  }

  NodeOps backwardOps() {
    return {
      NodeOp(Add(0.5f * (1.f / _1) * _2,
                 child(0)->grad(),
                 val_,
                 adj_))
    };
  }

  const std::string type() {
    return "sqrt";
  }

  virtual size_t hash() {
    if(!hash_) {
      size_t seed = NaryNodeOp::hash();
      boost::hash_combine(seed, epsilon_);
      hash_ = seed;
    }
    return hash_;
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
                     child(0)->val()))
    };
  }

  NodeOps backwardOps() {
    return {
      NodeOp(Add(2.f * _1 * _2,
                 child(0)->grad(),
                 child(0)->val(),
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
                     child(0)->val()))
    };
  }

  NodeOps backwardOps() {
    return {
      NodeOp(Add(-_1,
                 child(0)->grad(),
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
                      child(0)->val(),
                      indeces_))
    };
  }

  NodeOps backwardOps() {
    return {
      NodeOp(PasteRows(child(0)->grad(),
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

  virtual size_t hash() {
    if(!hash_) {
      size_t seed = NaryNodeOp::hash();
      for(auto i : indeces_)
        boost::hash_combine(seed, i);
      hash_ = seed;
    }
    return hash_;
  }


  std::vector<size_t> indeces_;
};

struct ColsNodeOp : public UnaryNodeOp {
  template <typename ...Args>
  ColsNodeOp(Expr a, const std::vector<size_t>& indeces, Args ...args)
    : UnaryNodeOp(a, keywords::shape=newShape(a, indeces), args...),
      indeces_(indeces) {
  }

  NodeOps forwardOps() {
    // @TODO: solve this with a tensor!

    return {
      NodeOp(CopyCols(val_,
                      child(0)->val(),
                      indeces_))
    };
  }

  NodeOps backwardOps() {
    return {
      NodeOp(PasteCols(child(0)->grad(),
                       adj_,
                       indeces_))
    };
  }

  template <class ...Args>
  Shape newShape(Expr a, const std::vector<size_t>& indeces) {
    Shape shape = a->shape();
    shape.set(1, indeces.size());
    return shape;
  }

  const std::string type() {
    return "cols";
  }

  const std::string color() {
    return "orange";
  }

  virtual size_t hash() {
    if(!hash_) {
      size_t seed = NaryNodeOp::hash();
      for(auto i : indeces_)
        boost::hash_combine(seed, i);
      hash_ = seed;
    }
    return hash_;
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
                       val_, child(0)->val()))
    };
  }

  NodeOps backwardOps() {
    return {
      NodeOp(Transpose(getCublasHandle(),
                       child(0)->grad(), adj_))
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

class ReshapeNodeOp : public UnaryNodeOp {
private:
  Expr reshapee_;

public:
  template <typename ...Args>
  ReshapeNodeOp(Expr a, Shape shape, Args ...args)
    : UnaryNodeOp(a, keywords::shape=shape, args...),
      reshapee_(a) { }

  
    
  size_t allocate() { return 0; }
  void free() {}

  void forward() {}
  void backward() {}

  void init_dependent() {
    reshapee_->init_dependent();
  }

  void set_zero_adjoint() {
    reshapee_->set_zero_adjoint();
  }

  Tensor& val()  {
    auto childVal = reshapee_->val();
    val_.reset(new TensorBase(childVal->data(), shape(), childVal->getDevice()));
    return val_;
  };

  Tensor& grad() {
    auto childGrad = reshapee_->grad();
    adj_.reset(new TensorBase(childGrad->data(), shape(), childGrad->getDevice()));
    return adj_;
  };

  const std::string type() {
    return "reshape";
  }

  const std::string color() {
    return "grey";
  }

  virtual size_t hash() {
    if(!hash_) {
      size_t seed = NaryNodeOp::hash();
      for(auto s : shape())
        boost::hash_combine(seed, s);
      hash_ = seed;
    }
    return hash_;
  }

};

class TimestepNodeOp : public UnaryNodeOp {
private:
  Expr stepNode_;
  size_t step_;

public:
  TimestepNodeOp(Expr a, size_t step)
    : UnaryNodeOp(a, keywords::shape=newShape(a)),
      stepNode_(a), step_(step)
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
    stepNode_->init_dependent();
  }

  void set_zero_adjoint() {
    stepNode_->set_zero_adjoint();
  }

  Tensor& val()  {
    auto childVal = stepNode_->val();
    size_t offset = step_ * shape().elements();
    val_.reset(new TensorBase(childVal->data() + offset, shape(), childVal->getDevice()));
    return val_;
  };

  Tensor& grad() {
    auto childGrad = stepNode_->grad();
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

  virtual size_t hash() {
    if(!hash_) {
       hash_ = NaryNodeOp::hash();
       boost::hash_combine(hash_, step_);
    }
    return hash_;
  }

};

struct ShiftNodeOp : public UnaryNodeOp {
  template <typename ...Args>
  ShiftNodeOp(Expr a, Shape shift, Args ...args)
    : UnaryNodeOp(a, keywords::shape=a->shape(), args...),
      shift_(shift) {
  }

  NodeOps forwardOps() {
    return {
      NodeOp(Shift(val_,
                   child(0)->val(),
                   shift_))
    };
  }

  NodeOps backwardOps() {
    return {
      NodeOp(Shift(child(0)->grad(),
                   adj_,
                   shift_,
                   true))
    };
  }

  const std::string type() {
    return "shift";
  }

  virtual size_t hash() {
    if(!hash_) {
      size_t seed = NaryNodeOp::hash();
      for(auto i : shape_)
        boost::hash_combine(seed, i);
      hash_ = seed;
    }
    return hash_;
  }


  Shape shift_;
};

struct LexicalProbNodeOp : public NaryNodeOp {
  template <typename ...Args>
  LexicalProbNodeOp(Expr logits, Expr att, float eps, Ptr<sparse::CSR> lf, Args ...args)
    : NaryNodeOp({logits, att}, keywords::shape=logits->shape(), args...),
      eps_(eps),
      lf_(lf) {
  }

  void forward() {
    sparse::LfaForward(val_,
                       child(0)->val(),
                       child(1)->val(),
                       lf_);
    // val = x + ln(p + eps)
    Element(_1 = (Log(_1 + eps_) + _2),
            val_, child(0)->val());
  }
  
  void backward() {
      Add(_1, child(0)->grad(), adj_);
      // adj' = adj / (p + eps) = adj / exp(val - x)
      Element(_1 = _1 / Exp(_2 - _3),
              adj_, val_, child(0)->val()); 
      sparse::LfaBackward(child(1)->grad(), adj_, lf_);
  }

  const std::string type() {
    return "lexical_prob";
  }

  virtual size_t hash() {
    if(!hash_) {
      size_t seed = NaryNodeOp::hash();
      boost::hash_combine(seed, (size_t)lf_.get());
      hash_ = seed;
    }
    return hash_;
  }

  float eps_;
  Ptr<sparse::CSR> lf_;
};

}
