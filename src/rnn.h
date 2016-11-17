#include <algorithm>
#include <chrono>
#include <iomanip>
#include <string>
#include <cstdio>
#include <boost/timer/timer.hpp>

#include "marian.h"
#include "node_operators_binary.h"
#include "expression_graph.h"

namespace marian {

struct ParametersTanh {
  Expr U, W, b;
  float dropout = 0;
};

class Tanh {
  public:
    Tanh(ParametersTanh params)
    : params_(params) {}

    Expr apply(Expr input, Expr state) {
      using namespace keywords;

      Expr output = dot(input, params_.W) + dot(state, params_.U);
      if(params_.b)
        output = output + params_.b;
      output = tanh(output);

      if(params_.dropout > 0)
        output = dropout(output, value=params_.dropout);

      return output;
    }

  private:
    ParametersTanh params_;
};

/***************************************************************/

struct ParametersGRU {
  Expr Uz, Wz, bz;
  Expr Ur, Wr, br;
  Expr Ux, Wx, bx;
  float dropout = 0;
};

class GRU {
  public:
    GRU(ParametersGRU params)
    : params_(params) {}

    Expr apply(Expr input, Expr state) {
      using namespace keywords;

      Expr z = dot(input, params_.Wz) + dot(state, params_.Uz);
      if(params_.bz)
        z = z + params_.bz;
      z = logit(z);

      Expr r = dot(input, params_.Wr) + dot(state, params_.Ur);
      if(params_.br)
        r = r + params_.br;
      r = logit(r);

      Expr h = dot(input, params_.Wx) + dot(state, params_.Ux) * r;
      if(params_.bx)
        h = h + params_.bx;
      h = tanh(h);

      // constant 1 in (1-z)*h+z*s
      auto one = state->graph()->ones(shape=state->shape());

      auto output = (one - z) * h + z * state;

      if(params_.dropout > 0)
        output = dropout(output, value=params_.dropout);

      return output;
    }

  private:
    ParametersGRU params_;
};

/***************************************************************/

struct ParametersGRUFast {
  Expr U, W, b;
  float dropout = 0;
};

struct GRUFastNodeOp : public NaryNodeOp {
  template <typename ...Args>
  GRUFastNodeOp(const std::vector<Expr>& nodes, Args ...args)
    : NaryNodeOp(nodes,
                 keywords::shape=nodes.front()->shape(),
                 args...) { }

  void forward() {
    std::vector<Tensor> inputs;
    inputs.reserve(children_.size());
    for(auto child : children_)
      inputs.push_back(child->val());

    GRUFastForward(val_, inputs);
  }

  void backward() {
    std::vector<Tensor> inputs;
    std::vector<Tensor> outputs;

    inputs.reserve(children_.size());
    outputs.reserve(children_.size());
    for(auto child : children_) {
      inputs.push_back(child->val());
      outputs.push_back(child->grad());
    }

    GRUFastBackward(outputs, inputs, adj_);
  }

  virtual std::string graphviz() {
    std::stringstream ss;
    ss << "\"" << this << "\" [shape=\"box\", label=" << label("GRUFast")
      << ", style=\"filled\", fillcolor=\"yellow\"]" << std::endl;
    for(auto child : children_)
      ss << "\"" << child << "\" -> \"" << this << "\"" << std::endl;
    ss << std::endl;
    return ss.str();
  }
};

Expr grufast(const std::vector<Expr>& nodes) {
  return Expression<GRUFastNodeOp>(nodes);
}

class GRUFast {
  public:
    GRUFast(ParametersGRUFast params)
    : params_(params) {}

    Expr apply(Expr input, Expr state) {
      using namespace keywords;

      auto xW = dot(input, params_.W);
      auto sU = dot(state, params_.U);

      auto output = grufast({state, xW, sU, params_.b});

      if(params_.dropout > 0)
        output = dropout(output, value=params_.dropout);

      return output;
    }

  private:
    ParametersGRUFast params_;
};

/***************************************************************/

template <class Cell = Tanh>
class RNN {
  public:

    template <class Parameters>
    RNN(const Parameters& params)
    : cell_(params) {}

    std::vector<Expr> apply(const std::vector<Expr>& inputs,
                            const Expr initialState) {
      return apply(inputs.begin(), inputs.end(),
                   initialState);
    }

    template <class Iterator>
    std::vector<Expr> apply(Iterator it, Iterator end,
                            const Expr initialState) {
      std::vector<Expr> outputs;
      auto state = initialState;
      while(it != end) {
        state = cell_.apply(*it++, state);
        outputs.push_back(state);
      }
      return outputs;
    }

  private:
    Cell cell_;
};

}
