#include "node.h"
#include "tensor_operators.h"

namespace marian {

struct UnaryNodeOp : public Node {
    ChainPtr a_;

    template <typename ...Args>
    UnaryNodeOp(ChainPtr a, Args ...args)
    : Node(keywords::shape=a->shape(), //@TODO: Check keywords?
           args...), a_(a) {}

    void backward_numeric(Float delta) {
      using namespace std;

      cerr << "UnaryNodeOp::" << typeid(*this).name() << "::backward_numeric()" << endl;

	  Tensor input = a_->val();
	  size_t totSize = GetTotalSize(input.shape());

	  std::vector<float> preCalcGrad(totSize);
	  thrust::copy(a_->grad().begin(), a_->grad().end(), preCalcGrad.begin());
	  output("preCalcGrad", preCalcGrad);

	  // use df/dx to calc grad
	  backward();
	  //cerr << "orig a_->grad()=" << a_->grad().Debug() << endl;

	  std::vector<float> diffGrad(totSize);
	  thrust::copy(a_->grad().begin(), a_->grad().end(), diffGrad.begin());
	  output("diffGrad", diffGrad);

	  // reset grad
	  thrust::copy(preCalcGrad.begin(), preCalcGrad.end(), a_->grad().begin());
	  //cerr << "reset a_->grad()=" << a_->grad().Debug() << endl;

	  // START CALC of numerical gradient
	  // new values
	  input.incr(delta);

	  forward();
	  //cerr << "input=" << input.Debug() << endl;
	  //cerr << "val_=" << val_.Debug() << endl;

	  std::vector<float> newVal(totSize);
	  thrust::copy(val_.begin(), val_.end(), newVal.begin());
	  //output("newVal", newVal);

	  // old values
	  input.incr(-delta);

	  forward();
	  //cerr << "input=" << input.Debug() << endl;
	  //cerr << "val_=" << val_.Debug() << endl;

	  std::vector<float> origVal(totSize);
	  thrust::copy(val_.begin(), val_.end(), origVal.begin());
	  //output("origVal", origVal);

	  // calc gradient
	  //cerr << "adj_=" << adj_.Debug() << endl;
	  std::vector<float> adjVec(totSize);
	  thrust::copy(adj_.begin(), adj_.end(), adjVec.begin());

	  std::vector<float> numericalGrad(totSize);
	  for (size_t i = 0; i < totSize; ++i) {
		  numericalGrad[i] = preCalcGrad[i] + (adjVec[i] * (newVal[i] - origVal[i]) / delta);
	  }
	  output("numericalGrad", numericalGrad);
	  //cerr << "numeric a_->grad()=" << a_->grad().Debug() << endl;

	  // set grad results
	  thrust::copy(numericalGrad.begin(), numericalGrad.end(), a_->grad().begin());

	  // print out diff between diffGrad and numericalGrad
	  std::vector<float> origGrad(totSize);
	  std::vector<float> diff(totSize);

	  thrust::copy(a_->grad().begin(), a_->grad().end(), origGrad.begin());
	  for (size_t i = 0; i < totSize; ++i) {
		  diff[i] = (diffGrad[i] - numericalGrad[i]) / delta;
	  }
	  output("diff", diff);
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

// @TODO: slow and probably buggy
struct DropoutNodeOp : public UnaryNodeOp {
  template <typename ...Args>
  DropoutNodeOp(Args ...args)
  : UnaryNodeOp(args...),
    p_(0.5), seed_(time(0)) { }

  void forward() {
    //Element(_1 = Bernoulli(p_, (size_t)this) * _2,
    //        val_, a_->val())
    Dropout(val_, a_->val(), p_, seed_++);
  }

  void backward() {
    Element(_1 += _2 * (_3 != 0.0f), // transform non-zero to 1
            a_->grad(), adj_, val_);
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
    int seed_;
};


struct SoftmaxNodeOp : public UnaryNodeOp {
  template <typename ...Args>
    SoftmaxNodeOp(Args ...args)
    : UnaryNodeOp(args...) { }

  void forward() {
    // B = softmax(A).
    thrust::copy(a_->val().begin(), a_->val().end(), val_.begin());
    // Safe version of softmax.
    Softmax(&val_);
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

