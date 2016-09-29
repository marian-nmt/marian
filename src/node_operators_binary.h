#include "node.h"
#include "tensor_operators.h"

namespace marian {

struct BinaryNodeOp : public Node {
  ChainPtr a_;
  ChainPtr b_;

  template <typename ...Args>
  BinaryNodeOp(ChainPtr a, ChainPtr b, Args ...args)
   : Node(args...), a_(a), b_(b),
     skipInference_(a->skipInference_ || b->skipInference_),
     skipTraining_(a->skipTraining_ || b->skipTraining_) { }

  void backward_debug(Float delta) {
	  using namespace std;

	  cerr << "BinaryNodeOp::" << typeid(*this).name() << "::backward_debug()" << endl;

	  std::vector<float> preCalcGradA, diffGradA, numericalGradA;
	  preCalcGradA << a_->grad();
	  //output("preCalcGradA", preCalcGradA);

	  std::vector<float> preCalcGradB, diffGradB, numericalGradB;
	  preCalcGradB << b_->grad();
	  //output("preCalcGradB", preCalcGradB);

	  // use df/dx to calc grad
	  backward();
	  cerr << "orig a_->grad()=" << a_->grad().Debug() << endl;
	  cerr << "orig b_->grad()=" << b_->grad().Debug() << endl;

	  diffGradA << a_->grad();
	  diffGradB << b_->grad();

	  //cerr << "orig a_->grad()=" << a_->grad().Debug() << endl;
	  //cerr << "orig b_->grad()=" << b_->grad().Debug() << endl;

	  cerr << "TENSOR A:" << endl;
	  a_->grad().set(preCalcGradA);
	  b_->grad().set(preCalcGradB);

	  calc_numeric_grad(delta, a_->val(), a_->grad());
	  cerr << "numerical a_->grad()=" << a_->grad().Debug() << endl;

	  numericalGradA << a_->grad();
	  outputL2Norm("TENSOR A", diffGradA, numericalGradA);


	  cerr << "TENSOR B:" << endl;
	  a_->grad().set(preCalcGradA);
	  b_->grad().set(preCalcGradB);

	  calc_numeric_grad(delta, b_->val(), b_->grad());
	  cerr << "numerical b_->grad()=" << b_->grad().Debug() << endl;

	  numericalGradB << b_->grad();
	  outputL2Norm("TENSOR B", diffGradB, numericalGradB);

	  // reset to diff grad
	  a_->grad().set(diffGradA);
	  b_->grad().set(diffGradB);
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
    ss << "\"" << this << "\" [shape=\"box\", label=" << label("•")
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
    ss << "\"" << this << "\" [shape=\"box\", label=" << label("x")
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

  // We're caching the logsoftmax probabilities here because we'll need them for
  // the backward computation.
  void forward() {
    // C = -dot(B, logsoftmax(A)).
    if (probs_) {
      probs_.set(0.0);
    } else {
      probs_.allocate(a_->val().shape(), 0.0);
    }

	CudnnLogSoftmax(probs_, a_->val());
	if(!result_)
	  result_.allocate(a_->val().shape());
    Element(_1 = -_2 * _3, result_, b_->val(), probs_);
    SumRowwise(result_, val_);
  }

  // @TODO: In most cases it's wasteful to compute the derivative with respect
  // to the second input which is typically an input node in the computation
  // graph. In general the backward functions can skip the computation of
  // gradients wrt input nodes.
  void backward() {
	// We are using logsoftmax for this and cached probs are logs.
    // For each row, the first input derivative is given by adj * (exp(p) - y),
    // where y is the gold label distribution (e.g. one hot vector) and
    // p is the softmax output (probabilities).
    // The second input derivative is -adj*p.

    // Compute first input derivative.
    Element(_1 += _2 * (Exp(_3) - _4),
			a_->grad(), adj_, probs_, b_->val());

    // Compute second input derivative.
    Element(_1 -= _2 * _3, b_->grad(),
			adj_, probs_);
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
