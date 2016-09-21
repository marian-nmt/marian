#include "node.h"
#include "tensor_operators.h"

namespace marian {

struct BinaryNodeOp : public Node {
  ChainPtr a_;
  ChainPtr b_;

  template <typename ...Args>
  BinaryNodeOp(ChainPtr a, ChainPtr b, Args ...args)
   : Node(args...), a_(a), b_(b) {}


  void backward_debug(Float delta) {
	  using namespace std;

	  cerr << "BinaryNodeOp::" << typeid(*this).name() << "::backward_numeric()" << endl;

	  std::vector<float> preCalcGradA = StoreTensorInVec(a_->grad());
	  //output("preCalcGradA", preCalcGradA);

	  std::vector<float> preCalcGradB = StoreTensorInVec(b_->grad());
	  //output("preCalcGradB", preCalcGradB);

	  // use df/dx to calc grad
	  backward();
	  //cerr << "orig a_->grad()=" << a_->grad().Debug() << endl;
	  //cerr << "orig b_->grad()=" << b_->grad().Debug() << endl;

	  cerr << "TENSOR A:" << endl;
	  calc_numeric_grad(delta, a_->val(), a_->grad(), preCalcGradA);
	  cerr << "TENSOR B:" << endl;
	  calc_numeric_grad(delta, b_->val(), b_->grad(), preCalcGradB);

  }


};

/*** Matrix Product ***/

struct DotNodeOp : public BinaryNodeOp {
  template <typename ...Args>
  DotNodeOp(ChainPtr a, ChainPtr b, Args ...args)
  : BinaryNodeOp(a, b,
                 keywords::shape=newShape(a, b),
                 args...) { }

  Shape newShape(ChainPtr a, ChainPtr b) {
    Shape shape1 = a->shape();
    Shape shape2 = b->shape();
    UTIL_THROW_IF2(shape1[1] != shape2[0],
                   "matrix product requires dimensions to match");
    shape1[1] = shape2[1];
    return shape1;
  }

  void forward() {
    // C = A*B
    Prod(val_, a_->val(), b_->val(), false, false);
  }

  void backward() {
    // D is the adjoint, the matrix of derivatives
    // df/dA += D*B.T
    // df/dB += A.T*D
    // beta set to 1.0 in gemm, C = dot(A,B) + beta * C
    // to sum gradients from different graph parts
    Prod(a_->grad(), adj_, b_->val(), false, true, 1.0);
    Prod(b_->grad(), a_->val(), adj_, true, false, 1.0);
  }

  virtual std::string graphviz() {
    std::stringstream ss;
    ss << "\"" << this << "\" [shape=\"box\", label=" << label("×")
      << ", style=\"filled\", fillcolor=\"orange\"]" << std::endl;
    ss << "\"" << a_ << "\" -> \"" << this << "\"" << std::endl;
    ss << "\"" << b_ << "\" -> \"" << this << "\"" << std::endl << std::endl;
    return ss.str();
  };

};

struct PlusNodeOp : public BinaryNodeOp {
  template <typename ...Args>
  PlusNodeOp(ChainPtr a, ChainPtr b, Args ...args)
    : BinaryNodeOp(a, b, keywords::shape=a->shape(), args...) { }

  void forward() {
    Element(_1 = _2 + _3,
            val_, a_->val(), b_->val());
  }

  void backward() {
    Element(_1 += _2,
            a_->grad(), adj_);
    Element(_1 += _2,
            b_->grad(), adj_);
  }

  virtual std::string graphviz() {
    std::stringstream ss;
    ss << "\"" << this << "\" [shape=\"box\", label=" << label("+")
      << ", style=\"filled\", fillcolor=\"yellow\"]" << std::endl;
    ss << "\"" << a_ << "\" -> \"" << this << "\"" << std::endl;
    ss << "\"" << b_ << "\" -> \"" << this << "\"" << std::endl << std::endl;
    return ss.str();
  };

};

struct ReLUPlusNodeOp : public BinaryNodeOp {
  template <typename ...Args>
  ReLUPlusNodeOp(ChainPtr a, ChainPtr b, Args ...args)
    : BinaryNodeOp(a, b, keywords::shape=a->shape(), args...) { }

  void forward() {
    Element(_1 = ReLU(_2 + _3),
            val_, a_->val(), b_->val());
  }

  void backward() {
    Element(_1 += _2 * ReLUback(_3 + _4),
            a_->grad(), adj_, a_->val(), b_->val());
    Element(_1 += _2 * ReLUback(_3 + _4),
            b_->grad(), adj_, a_->val(), b_->val());
  }

  virtual std::string graphviz() {
    std::stringstream ss;
    ss << "\"" << this << "\" [shape=\"box\", label=" << label("ReLU<br/>+")
      << ", style=\"filled\", fillcolor=\"yellow\"]" << std::endl;
    ss << "\"" << a_ << "\" -> \"" << this << "\"" << std::endl;
    ss << "\"" << b_ << "\" -> \"" << this << "\"" << std::endl << std::endl;
    return ss.str();
  };

};

struct MinusNodeOp : public BinaryNodeOp {
  template <typename ...Args>
  MinusNodeOp(ChainPtr a, ChainPtr b, Args ...args)
    : BinaryNodeOp(a, b, keywords::shape=a->shape(), args...) { }

  void forward() {
    Element(_1 = _2 - _3,
            val_, a_->val(), b_->val());
  }

  void backward() {
    Element(_1 += _2,
            a_->grad(), adj_);
    Element(_1 -= _2,
            b_->grad(), adj_);
  }

  virtual std::string graphviz() {
    std::stringstream ss;
    ss << "\"" << this << "\" [shape=\"box\", label=" << label("-")
      << ", style=\"filled\", fillcolor=\"yellow\"]" << std::endl;
    ss << "\"" << a_ << "\" -> \"" << this << "\"" << std::endl;
    ss << "\"" << b_ << "\" -> \"" << this << "\"" << std::endl << std::endl;
    return ss.str();
  };

};

struct MultNodeOp : public BinaryNodeOp {
  template <typename ...Args>
  MultNodeOp(ChainPtr a, ChainPtr b, Args ...args)
    : BinaryNodeOp(a, b, keywords::shape=a->shape(), args...) { }

  void forward() {
    Element(_1 = _2 * _3,
            val_, a_->val(), b_->val());
  }

  void backward() {
    Element(_1 += _2 * _3,
            a_->grad(), adj_, b_->val());
    Element(_1 += _2 * _3,
            b_->grad(), adj_, a_->val());
  }

  virtual std::string graphviz() {
    std::stringstream ss;
    ss << "\"" << this << "\" [shape=\"box\", label=" << label("•")
      << ", style=\"filled\", fillcolor=\"yellow\"]" << std::endl;
    ss << "\"" << a_ << "\" -> \"" << this << "\"" << std::endl;
    ss << "\"" << b_ << "\" -> \"" << this << "\"" << std::endl << std::endl;
    return ss.str();
  };

};

struct DivNodeOp : public BinaryNodeOp {
  template <typename ...Args>
  DivNodeOp(ChainPtr a, ChainPtr b, Args ...args)
    : BinaryNodeOp(a, b, keywords::shape=a->shape(), args...) { }

  void forward() {
    Element(_1 = _2 / _3,
            val_, a_->val(), b_->val());
  }

  void backward() {
    Element(_1 += _2 * 1.0f / _3,
            a_->grad(), adj_, b_->val());
    Element(_1 -= _2 * _3 / (_4 * _4),
            b_->grad(), adj_, a_->val(), b_->val());
  }

  virtual std::string graphviz() {
    std::stringstream ss;
    ss << "\"" << this << "\" [shape=\"box\", label=" << label("÷")
      << ", style=\"filled\", fillcolor=\"yellow\"]" << std::endl;
    ss << "\"" << a_ << "\" -> \"" << this << "\"" << std::endl;
    ss << "\"" << b_ << "\" -> \"" << this << "\"" << std::endl << std::endl;
    return ss.str();
  };

};

// Cross-entropy node. It computes -b*log(softmax(a)), summing rowwise.
struct CrossEntropyNodeOp : public BinaryNodeOp {
  template <typename ...Args>
    CrossEntropyNodeOp(ChainPtr a, ChainPtr b, Args ...args)
    : BinaryNodeOp(a, b,
                   keywords::shape=newShape(a, b),
                   args...) { }

  Shape newShape(ChainPtr a, ChainPtr b) {
    Shape shape1 = a->shape();
    Shape shape2 = b->shape();
    UTIL_THROW_IF2(shape1[0] != shape2[0] || shape1[1] != shape2[1],
                   "cross entropy requires dimensions to match");
    shape1[1] = 1;
    return shape1;
  }

  // We're caching the softmax probabilities here because we'll need them for
  // the backward computation.
  void forward() {
    // C = -dot(B, log(softmax(A))).
    if (probs_) {
      probs_.set(0.0);
    } else {
      probs_.allocate(a_->val().shape(), 0.0);
    }
    thrust::copy(a_->val().begin(), a_->val().end(), probs_.begin());
    Softmax(&probs_); // Safe version of softmax.
	if(!result_)
	  result_.allocate(a_->val().shape());
    Element(_1 = -_2 * Log(_3), result_, b_->val(), probs_);
    SumRowwise(result_, val_);
  }

  // @TODO: In most cases it's wasteful to compute the derivative with respect
  // to the second input which is typically an input node in the computation
  // graph. In general the backward functions can skip the computation of
  // gradients wrt input nodes.
  void backward() {
    // For each row, the first input derivative is given by adj * (p - y),
    // where y is the gold label distribution (e.g. one hot vector) and
    // p is the softmax output (probabilities).
    // The second input derivative is -adj*log(p).
    if(!result_)
	  result_.allocate(probs_.shape());

    // Compute first input derivative.
    Element(_1 = _2 -  _3, result_, probs_, b_->val());
    ScaleRowwise(result_, adj_);
    Element(_1 += _2, a_->grad(), result_);

    // Compute second input derivative.
    Element(_1 = -Log(_2), result_, probs_); // @TODO: use a cached log here.
    ScaleRowwise(result_, adj_);
    Element(_1 += _2, b_->grad(), result_);
  }

  virtual std::string graphviz() {
    std::stringstream ss;
    ss << "\"" << this << "\" [shape=\"box\", label=" << label("x-ent")
      << ", style=\"filled\", fillcolor=\"orange\"]" << std::endl;
    ss << "\"" << a_ << "\" -> \"" << this << "\"" << std::endl << std::endl;
    ss << "\"" << b_ << "\" -> \"" << this << "\"" << std::endl << std::endl;
    return ss.str();
  };

 protected:
  Tensor probs_;
  Tensor result_;
};


}

