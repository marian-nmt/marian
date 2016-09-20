#include "node.h"
#include "tensor_operators.h"

namespace marian {

struct BinaryNodeOp : public Node {
  ChainPtr a_;
  ChainPtr b_;

  template <typename ...Args>
  BinaryNodeOp(ChainPtr a, ChainPtr b, Args ...args)
   : Node(args...), a_(a), b_(b) {}

  std::vector<float> StoreTensorInVec(Tensor tensor)
  {
	  size_t totSize = GetTotalSize(tensor.shape());
	  std::vector<float> vec(totSize);
	  thrust::copy(tensor.begin(), tensor.end(), vec.begin());
	  return vec;
  }

  void calc_numeric_grad(
		  Float delta,
		  Tensor input,
		  Tensor grad,
		  const std::vector<float> &prevCalcGrad
		  )
  {
	  size_t totSize = GetTotalSize(input.shape());

	  std::vector<float> diffGrad(totSize);
	  thrust::copy(grad.begin(), grad.end(), diffGrad.begin());
	  output("diffGrad", diffGrad);

	  // reset grad
	  thrust::copy(prevCalcGrad.begin(), prevCalcGrad.end(), grad.begin());
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
		  numericalGrad[i] = prevCalcGrad[i] + (adjVec[i] * (newVal[i] - origVal[i]) / delta);
	  }
	  output("numericalGrad", numericalGrad);
	  //cerr << "numeric a_->grad()=" << a_->grad().Debug() << endl;

	  // set grad results
	  thrust::copy(numericalGrad.begin(), numericalGrad.end(), grad.begin());

	  // print out diff between diffGrad and numericalGrad
	  std::vector<float> origGrad(totSize);
	  std::vector<float> diff(totSize);

	  thrust::copy(grad.begin(), grad.end(), origGrad.begin());
	  for (size_t i = 0; i < totSize; ++i) {
		  diff[i] = (diffGrad[i] - numericalGrad[i]) / delta;
	  }
	  output("diff", diff);

  }

  void backward_numeric(Float delta) {
	  using namespace std;

	  cerr << "BinaryNodeOp::" << typeid(*this).name() << "::backward_numeric()" << endl;

	  std::vector<float> preCalcGradA = StoreTensorInVec(a_->grad());
	  //output("preCalcGradA", preCalcGradA);

	  std::vector<float> preCalcGradB = StoreTensorInVec(b_->grad());
	  //output("preCalcGradB", preCalcGradB);

	  // use df/dx to calc grad
	  backward();
	  //cerr << "orig a_->grad()=" << a_->grad().Debug() << endl;

	  cerr << "TENSOR A:" << endl;
	  calc_numeric_grad(delta, a_->val(), a_->grad(), preCalcGradA);
	  cerr << "TENSOR B:" << endl;
	  calc_numeric_grad(delta, b_->val(), b_->grad(), preCalcGradB);

	  // redo proper grad
	  backward();
  }

  void output(const std::string &title, const std::vector<float> &vec)
  {
	  std::cerr << title << " " << vec.size() << ":";
	  for (size_t i = 0; i < vec.size(); ++i) {
		  std::cerr << vec[i] << " ";
	  }
	  std::cerr << std::endl;
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
    Tensor result(a_->val().shape());
    Element(_1 = -_2 * Log(_3), result, b_->val(), probs_);
    SumRowwise(result, val_);
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
    Tensor result(probs_.shape());

    // Compute first input derivative.
    Element(_1 = _2 -  _3, result, probs_, b_->val());
    ScaleRowwise(result, adj_);
    Element(_1 += _2, a_->grad(), result);

    // Compute second input derivative.
    Element(_1 = -Log(_2), result, probs_); // @TODO: use a cached log here.
    ScaleRowwise(result, adj_);
    Element(_1 += _2, b_->grad(), result);
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

};


}

