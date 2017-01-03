#pragma once

#include "node.h"
#include "tensors/tensor.h"
#include "tensor_operators.h"
#include "thrust_functions.h"

namespace marian {

struct UnaryNodeOp : public Node {
    Expr a_;

    template <typename ...Args>
    UnaryNodeOp(Expr a, Args ...args)
    : Node(a->graph(),
           keywords::shape=a->shape(), //@TODO: Check keywords?
           keywords::no_inference=a->skipped_inference() || keywords::Get(keywords::no_inference, false, args...),
           keywords::no_training=a->skipped_training() || keywords::Get(keywords::no_training, false, args...),
           args...),
        a_(a)
    {
        remove_children_from_top_nodes();
    }

    ~UnaryNodeOp() {}

    void remove_children_from_top_nodes();

    void backward_debug(Float delta) {
        using namespace std;
        //
        //cerr << "UnaryNodeOp::" << typeid(*this).name() << "::backward_numeric()" << endl;
        //
        //std::vector<float> preCalcGradA, diffGradA, numericalGradA;
        //a_->grad() >> preCalcGradA ;
        ////output("preCalcGradA", preCalcGradA);
        //
        //// use df/dx to calc grad
        //backward();
        //cerr << "orig a_->grad()=" << a_->grad().Debug() << endl;
        //a_->grad() >> diffGradA;
        //
        //a_->grad()->set(preCalcGradA);
        //
        //calc_numeric_grad(delta, a_->val(), a_->grad());
        ////cerr << "numerical a_->grad()=" << a_->grad()->Debug() << endl;
        //
        //a_->grad() >> numericalGradA;
        //
        //outputL2Norm("", diffGradA, numericalGradA);
        //
        //// reset to diff grad
        //a_->grad()->set(diffGradA);
    }

};

struct LogitNodeOp : public UnaryNodeOp {
  template <typename ...Args>
  LogitNodeOp(Args ...args)
  : UnaryNodeOp(args...) {  }

  void forward() {
    Element(_1 = Sigma(_2),
            val_, a_->val());
  }

  void backward() {
    Element(_1 += _2 * _3 * (1.0f - _3),
            a_->grad(), adj_, val_);
  }

  virtual std::string graphviz() {
    std::stringstream ss;
    ss << "\"" << this << "\" [shape=\"box\", label=" << label("logit")
      << ", style=\"filled\", fillcolor=\"yellow\"]" << std::endl;
    ss << "\"" << a_ << "\" -> \"" << this << "\"" << std::endl << std::endl;
    return ss.str();
  };

};

struct TanhNodeOp : public UnaryNodeOp {
  template <typename ...Args>
  TanhNodeOp(Args ...args)
  : UnaryNodeOp(args...) { }

  void forward() {
    Element(_1 = Tanh(_2),
            val_, a_->val());
  }

  void backward() {
    Element(_1 += _2 * (1.0f - (_3 * _3)),
            a_->grad(), adj_, val_);
  }

  virtual std::string graphviz() {
    std::stringstream ss;
    ss << "\"" << this << "\" [shape=\"box\", label=" << label("tanh")
      << ", style=\"filled\", fillcolor=\"yellow\"]" << std::endl;
    ss << "\"" << a_ << "\" -> \"" << this << "\"" << std::endl << std::endl;
    return ss.str();
  };

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

  void forward() {
    Element(_1 = ReLU(_2),
            val_, a_->val());
  }

  void backward() {
    Element(_1 += _2 * ReLUback(_3),
            a_->grad(), adj_, a_->val());
  }

  virtual std::string graphviz() {
    std::stringstream ss;
    ss << "\"" << this << "\" [shape=\"box\", label=" << label("ReLU")
      << ", style=\"filled\", fillcolor=\"yellow\"]" << std::endl;
    ss << "\"" << a_ << "\" -> \"" << this << "\"" << std::endl << std::endl;
    return ss.str();
  };

};

/**
 * @brief Represents a <a href="https://en.wikipedia.org/wiki/Dropout_(neural_networks)">dropout</a> node
 *        in an expression graph.
 *
 * @see \cite dropout
 * @see \cite cudnn
 */
struct DropoutNodeOp : public UnaryNodeOp {
  template <typename ...Args>
  DropoutNodeOp(Args ...args)
  : UnaryNodeOp(args...),
    allocated_(false), p_(Get(keywords::value, 0.5)) {}

  ~DropoutNodeOp() {
    if(allocated_)
      CudnnDropoutDestroy(dropDesc_, space_, states_);
 }

  void inference() {
    Element(_1 = _2, val_, a_->val());
  }

  void forward() {
    if(!allocated_) {
        CudnnDropoutPrepare(a_->val(), p_,
                            &dropDesc_,
                            &space_, &spaceSize_,
                            &states_, (size_t)this); // seeding with pointer address
        allocated_ = true;
    }

    CudnnDropoutForward(dropDesc_, space_, spaceSize_,
                        val_, a_->val());
  }

  void backward() {
    CudnnDropoutBackward(dropDesc_, space_, spaceSize_,
                         a_->grad(), adj_);
  }

  virtual std::string graphviz() {
    std::stringstream ss;
    ss << "\"" << this << "\" [shape=\"box\", label=" << label("dropout")
      << ", style=\"filled\", fillcolor=\"yellow\"]" << std::endl;
    ss << "\"" << a_ << "\" -> \"" << this << "\"" << std::endl << std::endl;
    return ss.str();
  };

  private:
    bool allocated_;
    float p_;
    void* states_;
    void* space_;
    size_t spaceSize_;
    cudnnDropoutDescriptor_t dropDesc_;
};

struct SoftmaxNodeOp : public UnaryNodeOp {
  template <typename ...Args>
    SoftmaxNodeOp(Args ...args)
    : UnaryNodeOp(args...) { }

  void forward() {
    Softmax(val_, a_->val());
  }

  void backward() {
    // For each row, the Jacobian times vector is given by:
    // J * dy = p .* (dy - avg*1)
    // where avg = p'*dy and p is the softmax output (probabilities).
    //
    // For more information, see sec. 2.5 of the following reference:
    // André F. T. Martins and Ramon Astudillo.
    // "From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label
    // Classification." ICML 2016.
    // http://jmlr.org/proceedings/papers/v48/martins16.pdf

    SoftmaxGrad(a_->grad(), adj_, val_);
  }

  virtual std::string graphviz() {
    std::stringstream ss;
    ss << "\"" << this << "\" [shape=\"box\", label=" << label("softmax")
      << ", style=\"filled\", fillcolor=\"yellow\"]" << std::endl;
    ss << "\"" << a_ << "\" -> \"" << this << "\"" << std::endl << std::endl;
    return ss.str();
  };
};

struct LogSoftmaxNodeOp : public UnaryNodeOp {
  template <typename ...Args>
    LogSoftmaxNodeOp(Args ...args)
    : UnaryNodeOp(args...) { }

  void forward() {
    CudnnLogSoftmax(val_, a_->val());
  }

  void backward() {
    // Based on the description for softmax, we have logsoftmax:
    // J * dy = dy - avg*1
    // where avg = exp(p)'*dy and p is the softmax output (probabilities).
    LogSoftmaxGrad(a_->grad(), adj_, val_);
  }

  virtual std::string graphviz() {
    std::stringstream ss;
    ss << "\"" << this << "\" [shape=\"box\", label=" << label("log-softmax")
      << ", style=\"filled\", fillcolor=\"yellow\"]" << std::endl;
    ss << "\"" << a_ << "\" -> \"" << this << "\"" << std::endl << std::endl;
    return ss.str();
  };
};


struct ArgmaxNodeOp : public UnaryNodeOp {
  template <typename ...Args>
  ArgmaxNodeOp(Expr a, Args ...args)
    : UnaryNodeOp(a, keywords::shape=newShape(a), args...) { }

  void forward() {
    // B = softmax(A).
    //Argmax(&val_, &a_->val());
  }

  void backward() {
  }

  Shape newShape(Expr a) {
    Shape shape = a->shape();
    shape.set(0, 1);
    return shape;
  }


  virtual std::string graphviz() {
    std::stringstream ss;
    ss << "\"" << this << "\" [shape=\"box\", label="
      << label("argmax") << ", style=\"filled\", fillcolor=\"yellow\"]" << std::endl;
    ss << "\"" << a_ << "\" -> \"" << this << "\"" << std::endl << std::endl;
    return ss.str();
  };

};

struct SumNodeOp : public UnaryNodeOp {
  template <typename ...Args>
  SumNodeOp(Expr a, Args ...args)
    : UnaryNodeOp(a, keywords::shape=newShape(a, args...), args...) { }

  void forward() {
    Reduce(_1, val_, a_->val());
  }

  void backward() {
    Element(_1 += _2, a_->grad(), adj_);
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

  virtual std::string graphviz() {
    std::stringstream ss;
    ss << "\"" << this << "\" [shape=\"box\", label="
      << label("sum") << ", style=\"filled\", fillcolor=\"orange\"]" << std::endl;
    ss << "\"" << a_ << "\" -> \"" << this << "\"" << std::endl << std::endl;
    return ss.str();
  };

};

struct MeanNodeOp : public UnaryNodeOp {
  template <typename ...Args>
  MeanNodeOp(Expr a, Args ...args)
    : UnaryNodeOp(a, keywords::shape=newShape(a, args...), args...) { }

  void forward() {
    int left = a_->shape().elements() / val_->shape().elements();
    float scale = 1.f / left;
    Reduce(_1 * scale, val_, a_->val());
  }

  void backward() {
    int left = a_->shape().elements() / val_->shape().elements();
    float scale = 1.f / left;
    Element(_1 += _2 * scale, a_->grad(), adj_);
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

  virtual std::string graphviz() {
    std::stringstream ss;
    ss << "\"" << this << "\" [shape=\"box\", label="
      << label("mean") << ", style=\"filled\", fillcolor=\"orange\"]" << std::endl;
    ss << "\"" << a_ << "\" -> \"" << this << "\"" << std::endl << std::endl;
    return ss.str();
  }
};


struct LogNodeOp : public UnaryNodeOp {
  template <typename ...Args>
  LogNodeOp(Args ...args)
  : UnaryNodeOp(args...) {}

  void forward() {
    Element(_1 = Log(_2), val_, a_->val());
  }

  void backward() {
    Element(_1 += _2 * (1.f / _3),
            a_->grad(), adj_, a_->val());
  }

  virtual std::string graphviz() {
    std::stringstream ss;
    ss << "\"" << this << "\" [shape=\"box\", label="
      << label("log") << ", style=\"filled\", fillcolor=\"yellow\"]" << std::endl;
    ss << "\"" << a_ << "\" -> \"" << this << "\"" << std::endl << std::endl;
    return ss.str();
  };

};

struct ExpNodeOp : public UnaryNodeOp {
  template <typename ...Args>
    ExpNodeOp(Args ...args)
    : UnaryNodeOp(args...) { }

  void forward() {
    Element(_1 = Exp(_2), val_, a_->val());
  }

  void backward() {
    Element(_1 += _2 * Exp(_3),
            a_->grad(), adj_, a_->val());
  }

  virtual std::string graphviz() {
    std::stringstream ss;
    ss << "\"" << this << "\" [shape=\"box\", label=" << label("exp")
      << ", style=\"filled\", fillcolor=\"yellow\"]" << std::endl;
    ss << "\"" << a_ << "\" -> \"" << this << "\"" << std::endl << std::endl;
    return ss.str();
  };

};

struct NegNodeOp : public UnaryNodeOp {
  template <typename ...Args>
  NegNodeOp(Args ...args)
  : UnaryNodeOp(args...) { }

  void forward() {
    Element(_1 = -_2, val_, a_->val());
  }

  void backward() {
    Element(_1 += -_2, a_->grad(), adj_);
  }

  virtual std::string graphviz() {
    std::stringstream ss;
    ss << "\"" << this << "\" [shape=\"box\", label="
      << label("-") << ", style=\"filled\", fillcolor=\"yellow\"]" << std::endl;
    ss << "\"" << a_ << "\" -> \"" << this << "\"" << std::endl << std::endl;
    return ss.str();
  }
};

struct RowsNodeOp : public UnaryNodeOp {
  template <typename ...Args>
  RowsNodeOp(Expr a, const DeviceVector<size_t>& indeces, Args ...args)
    : UnaryNodeOp(a, keywords::shape=newShape(a, indeces), args...),
      indeces_(indeces) { }

  void forward() {
    CopyRows(val_, a_->val(), indeces_);
  }

  void backward() {
    PasteRows(a_->grad(), adj_, indeces_);
  }

  template <class ...Args>
  Shape newShape(Expr a, const DeviceVector<size_t>& indeces) {
    Shape shape = a->shape();
    shape.set(0, indeces.size());
    return shape;
  }

  virtual std::string graphviz() {
    std::stringstream ss;
    ss << "\"" << this << "\" [shape=\"box\", label="
      << label("rows") << ", style=\"filled\", fillcolor=\"orange\"]" << std::endl;
    ss << "\"" << a_ << "\" -> \"" << this << "\"" << std::endl << std::endl;
    return ss.str();
  }

  const DeviceVector<size_t> &indeces_;
};

struct TransposeNodeOp : public UnaryNodeOp {
  template <typename ...Args>
  TransposeNodeOp(Expr a, Args ...args)
    : UnaryNodeOp(a, keywords::shape=newShape(a), args...) { }

  void forward() {
    Transpose(val_, a_->val());
  }

  void backward() {
    Transpose(a_->grad(), adj_);
  }

  template <class ...Args>
  Shape newShape(Expr a) {
    Shape shape = a->shape();
    int temp = shape[0];
    shape.set(0, shape[1]);
    shape.set(1, temp);
    return shape;
  }

  virtual std::string graphviz() {
    std::stringstream ss;
    ss << "\"" << this << "\" [shape=\"box\", label="
      << label("transpose") << ", style=\"filled\", fillcolor=\"orange\"]" << std::endl;
    ss << "\"" << a_ << "\" -> \"" << this << "\"" << std::endl << std::endl;
    return ss.str();
  }
};

struct ReshapeNodeOp : public UnaryNodeOp {
  template <typename ...Args>
  ReshapeNodeOp(Expr a, Shape shape, Args ...args)
    : UnaryNodeOp(a, keywords::shape=shape, args...) { }

  size_t allocate(size_t batchSize) { return 0; }

  void forward() {}
  void backward() {}
  void init_dependent() {}
  void set_zero_adjoint() {}

  Tensor& val()  {
    val_.reset(new TensorGPU(a_->val()->data(), shape()));
    return val_;
  };

  Tensor& grad() {
    adj_.reset(new TensorGPU(a_->grad()->data(), shape()));
    return adj_;
  };


  virtual std::string graphviz() {
    std::stringstream ss;
    ss << "\"" << this << "\" [shape=\"box\", label="
      << label("reshape") << ", style=\"filled\", fillcolor=\"yellow\"]" << std::endl;
    ss << "\"" << a_ << "\" -> \"" << this << "\"" << std::endl << std::endl;
    return ss.str();
  }
};

//struct CrossEntropyPickNodeOp : public UnaryNodeOp {
//  template <typename ...Args>
//    CrossEntropyPickNodeOp(Expr a,
//                           const DeviceVector<size_t>& picks,
//                           Args ...args)
//    : UnaryNodeOp(a,
//                  keywords::shape=newShape(a),
//                  args...),
//      picks_(picks) { }
//
//  Shape newShape(Expr a) {
//    Shape shape1 = a->shape();
//    shape1.set(1, 1);
//    return shape1;
//  }
//
//  void forward();
//
//  void backward() {
//    // We are using logsoftmax for this and cached probs are logs.
//    // For each row, the first input derivative is given by adj * (exp(p) - y),
//    // where y is the gold label distribution (e.g. one hot vector) and
//    // p is the softmax output (probabilities).
//    // The second input derivative is -adj*p.
//
//    // Compute first input derivative.
//    Pick(_1 += _2 * (Exp(_3) - _4),
//         a_->grad(), adj_, probs_, picks_);
//  }
//
//  virtual std::string graphviz() {
//    std::stringstream ss;
//    ss << "\"" << this << "\" [shape=\"box\", label=" << label("x-ent")
//      << ", style=\"filled\", fillcolor=\"orange\"]" << std::endl;
//    ss << "\"" << a_ << "\" -> \"" << this << "\"" << std::endl << std::endl;
//    return ss.str();
//  };
//
// protected:
//  const DeviceVector<size_t> picks_;
//  Tensor probs_;
//};



}
