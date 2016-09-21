#include "node.h"
#include "tensor_operators.h"
#include "dropout.h"

namespace marian {

struct UnaryNodeOp : public Node {
    ChainPtr a_;

    template <typename ...Args>
    UnaryNodeOp(ChainPtr a, Args ...args)
    : Node(keywords::shape=a->shape(), //@TODO: Check keywords?
           args...), a_(a) {}

    void backward_debug(Float delta) {
      using namespace std;

      cerr << "UnaryNodeOp::" << typeid(*this).name() << "::backward_numeric()" << endl;

	  std::vector<float> preCalcGradA = StoreTensorInVec(a_->grad());
	  //output("preCalcGradA", preCalcGradA);

	  // use df/dx to calc grad
	  backward();
	  //cerr << "orig a_->grad()=" << a_->grad().Debug() << endl;

	  calc_numeric_grad(delta, a_->val(), a_->grad(), preCalcGradA);
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

  void check() {

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

// Scaling droput
struct DropoutNodeOp : public UnaryNodeOp {
  template <typename ...Args>
  DropoutNodeOp(Args ...args)
  : UnaryNodeOp(args...),
    p_(Get<float>(keywords::value, 0.5)) {}

  ~DropoutNodeOp() {
    if(bernoulli)
      bernoulli->FreeStates(states_);
  }
  
  void inference() {
    Element(_1 = _2, val_, a_->val());
  }
  
  void forward() {
    if(!bernoulli) {
      bernoulli.reset(new Bernoulli(p_, val_.shape()));
      bernoulli->InitStates(states_);
    }
    
    if(!mask_)
      mask_.allocate(val_.shape());
    
    auto f = [] __device__ (float& mask, float drop) {
      return mask = drop;
    };  
    Element(f, mask_, *bernoulli);
    Element(_1 = _2 * _3, val_, mask_, a_->val());
  }
  
  void backward() {    
    Element(_1 += _2 * _3, a_->grad(), adj_, mask_);
  }

  virtual std::string graphviz() {
    std::stringstream ss;
    ss << "\"" << this << "\" [shape=\"box\", label=" << label("dropout")
      << ", style=\"filled\", fillcolor=\"yellow\"]" << std::endl;
    ss << "\"" << a_ << "\" -> \"" << this << "\"" << std::endl << std::endl;
    return ss.str();
  };

  private:
    float p_;
    curandState* states_;
    std::shared_ptr<Bernoulli> bernoulli;
    Tensor mask_;
};


struct SoftmaxNodeOp : public UnaryNodeOp {
  template <typename ...Args>
    SoftmaxNodeOp(Args ...args)
    : UnaryNodeOp(args...) { }

  void forward() {
    CudnnSoftmax(val_, a_->val());
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
    CudnnLogSoftmaxGrad(a_->grad(), adj_, val_);
    //LogSoftmaxGrad(a_->grad(), adj_, val_);
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
  ArgmaxNodeOp(ChainPtr a, Args ...args)
    : UnaryNodeOp(a, keywords::shape=newShape(a), args...) { }

  void forward() {
    // B = softmax(A).
    Argmax(&val_, &a_->val());
  }

  void backward() {
  }

  Shape newShape(ChainPtr a) {
    Shape shape = a->shape();
    shape[1] = 1;
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
  };

};


}

