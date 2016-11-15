#pragma once

#include "node.h"
#include "thrust_functions.h"
#include "tensor_operators.h"

namespace marian {

struct BinaryNodeOp : public Node {
  Expr a_;
  Expr b_;

  template <typename ...Args>
  BinaryNodeOp(Expr a, Expr b, Args ...args)
   : Node(a->graph(),
      keywords::shape=keywords::Get(keywords::shape, a->shape(), args...),
      keywords::no_inference=a->skipped_inference()
      || b->skipped_inference()
      || keywords::Get(keywords::no_inference, false, args...),
          keywords::no_training=a->skipped_training()
      || b->skipped_training()
      || keywords::Get(keywords::no_training, false, args...),
      args...), a_(a), b_(b)
  {
  remove_children_from_top_nodes();
  }

  ~BinaryNodeOp() {}

  void remove_children_from_top_nodes();

  void backward_debug(Float delta) {
    //using namespace std;
    //
    //cerr << "BinaryNodeOp::" << typeid(*this).name() << "::backward_debug()" << endl;
    //
    //std::vector<float> preCalcGradA, diffGradA, numericalGradA;
    //preCalcGradA << a_->grad();
    ////output("preCalcGradA", preCalcGradA);
    //
    //std::vector<float> preCalcGradB, diffGradB, numericalGradB;
    //preCalcGradB << b_->grad();
    ////output("preCalcGradB", preCalcGradB);
    //
    //// use df/dx to calc grad
    //backward();
    //cerr << "orig a_->grad()=" << a_->grad().Debug() << endl;
    //cerr << "orig b_->grad()=" << b_->grad().Debug() << endl;
    //
    //diffGradA << a_->grad();
    //diffGradB << b_->grad();
    //
    ////cerr << "orig a_->grad()=" << a_->grad().Debug() << endl;
    ////cerr << "orig b_->grad()=" << b_->grad().Debug() << endl;
    //
    //cerr << "TENSOR A:" << endl;
    //a_->grad().set(preCalcGradA);
    //b_->grad().set(preCalcGradB);
    //
    //calc_numeric_grad(delta, a_->val(), a_->grad());
    //cerr << "numerical a_->grad()=" << a_->grad().Debug() << endl;
    //
    //numericalGradA << a_->grad();
    //outputL2Norm("TENSOR A", diffGradA, numericalGradA);
    //
    //
    //cerr << "TENSOR B:" << endl;
    //a_->grad().set(preCalcGradA);
    //b_->grad().set(preCalcGradB);
    //
    //calc_numeric_grad(delta, b_->val(), b_->grad());
    //cerr << "numerical b_->grad()=" << b_->grad().Debug() << endl;
    //
    //numericalGradB << b_->grad();
    //outputL2Norm("TENSOR B", diffGradB, numericalGradB);
    //
    //// reset to diff grad
    //a_->grad().set(diffGradA);
    //b_->grad().set(diffGradB);
  }


};

/**
 * @brief Represents a node in an expression graph capable of performing
 *        <a href="https://en.wikipedia.org/wiki/Matrix_multiplication#Matrix_product_.28two_matrices.29">matrix
 *        multiplication</a> of two input matrices.
 */
struct DotNodeOp : public BinaryNodeOp {
  template <typename ...Args>
  DotNodeOp(Expr a, Expr b, Args ...args)
  : BinaryNodeOp(a, b,
                 keywords::shape=newShape(a, b),
                 args...) { }

  Shape newShape(Expr a, Expr b) {
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
  PlusNodeOp(Args ...args)
    : BinaryNodeOp(args...) { }

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
  ReLUPlusNodeOp(Args ...args)
    : BinaryNodeOp(args...) { }

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
  MinusNodeOp(Args ...args)
    : BinaryNodeOp(args...) { }

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
  MultNodeOp(Args ...args)
    : BinaryNodeOp(args...) { }

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
    ss << "\"" << this << "\" [shape=\"box\", label=" << label("×")
      << ", style=\"filled\", fillcolor=\"yellow\"]" << std::endl;
    ss << "\"" << a_ << "\" -> \"" << this << "\"" << std::endl;
    ss << "\"" << b_ << "\" -> \"" << this << "\"" << std::endl << std::endl;
    return ss.str();
  };

};

struct DivNodeOp : public BinaryNodeOp {
  template <typename ...Args>
  DivNodeOp(Args ...args)
    : BinaryNodeOp(args...) { }

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
    CrossEntropyNodeOp(Expr a, Expr b, Args ...args)
    : BinaryNodeOp(a, b,
                   keywords::shape=newShape(a, b),
                   args...) { }

  Shape newShape(Expr a, Expr b) {
    Shape shape1 = a->shape();
    Shape shape2 = b->shape();
    UTIL_THROW_IF2(shape1[0] != shape2[0] || shape1[1] != shape2[1],
                   "cross entropy requires dimensions to match");
    shape1[1] = 1;
    return shape1;
  }

  void forward();

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
    Element(_1 -= _2 * _3,
      b_->grad(), adj_, probs_);
  }

  virtual std::string graphviz() {
    std::stringstream ss;
    ss << "\"" << this << "\" [shape=\"box\", label=" << label("x-ent")
      << ", style=\"filled\", fillcolor=\"orange\"]" << std::endl;
    ss << "\"" << a_ << "\" -> \"" << this << "\"" << std::endl;
    ss << "\"" << b_ << "\" -> \"" << this << "\"" << std::endl << std::endl;
    return ss.str();
  };

 protected:
  Tensor probs_;
  Tensor result_;
};

// an n-ary node

struct NaryNodeOp : public Node {
  std::vector<Expr> children_;

  template <typename ...Args>
  NaryNodeOp(const std::vector<Expr>& nodes, Args ...args)
   : Node(nodes.back()->graph(),
      keywords::shape=keywords::Get(keywords::shape, nodes.back()->shape(), args...),
      keywords::no_inference=
      std::any_of(nodes.begin(), nodes.end(),
                  [](Expr a) { return a->skipped_inference(); })
      || keywords::Get(keywords::no_inference, false, args...),
      keywords::no_training=
      std::any_of(nodes.begin(), nodes.end(),
                  [](Expr a) { return a->skipped_training(); })
      || keywords::Get(keywords::no_training, false, args...),
      args...), children_(nodes)
  {
    remove_children_from_top_nodes();
  }

  ~NaryNodeOp() {}

  void remove_children_from_top_nodes();

  void backward_debug(Float delta) {}

};

struct ConcatenateNodeOp : public NaryNodeOp {
  template <typename ...Args>
  ConcatenateNodeOp(const std::vector<Expr>& nodes, Args ...args)
    : NaryNodeOp(nodes,
                 keywords::shape=newShape(nodes, args...),
                 args...) { }

  template <typename ...Args>
  Shape newShape(const std::vector<Expr>& nodes, Args ...args) {
    Shape shape = nodes.back()->shape();
    shape[0] = 0;
    for(auto child : nodes)
      shape[0] += child->shape()[0];
    return shape;
  }

  void forward() {
    std::vector<Tensor> concatenees;
    for(auto child : children_)
      concatenees.push_back(child->val());
    Concatenate(val_, concatenees);
  }

  void backward() {
    std::vector<Tensor> deconcatenees;
    for(auto child : children_)
      deconcatenees.push_back(child->grad());
    Deconcatenate(deconcatenees, adj_);
  }

  virtual std::string graphviz() {
    std::stringstream ss;
    ss << "\"" << this << "\" [shape=\"box\", label=" << label("Concat")
      << ", style=\"filled\", fillcolor=\"orange\"]" << std::endl;
    for(auto child : children_)
      ss << "\"" << child << "\" -> \"" << this << "\"" << std::endl;
    ss << std::endl;
    return ss.str();
  };

};


}
