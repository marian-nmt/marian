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
  bool final_;

  template <typename ...Args>
  GRUFastNodeOp(const std::vector<Expr>& nodes, bool final, Args ...args)
    : NaryNodeOp(nodes,
                 keywords::shape=nodes.front()->shape(),
                 args...),
      final_(final) { }

  void forward() {
    std::vector<Tensor> inputs;
    for(auto child : children_)
      inputs.push_back(child->val());

    GRUFastForward(val_, inputs, final_);
  }

  void backward() {
    std::vector<Tensor> inputs;
    std::vector<Tensor> outputs;
    for(auto child : children_) {
      inputs.push_back(child->val());
      outputs.push_back(child->grad());
    }

    GRUFastBackward(outputs, inputs, adj_, final_);
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

Expr grufast(const std::vector<Expr>& nodes, bool final = false) {
  return Expression<GRUFastNodeOp>(nodes, final);
}

class GRUFast {
  public:
    GRUFast(ParametersGRUFast params, Expr dropoutMask = nullptr)
    : params_(params),
      dropoutMask_(dropoutMask) {}

    Expr apply(Expr input, Expr state, Expr mask = nullptr) {
      auto xW = dot(input, params_.W);
      auto sU = dot(state, params_.U);

      auto output = mask ?
        grufast({state, xW, sU, params_.b, mask}) :
        grufast({state, xW, sU, params_.b});

      if(dropoutMask_)
        output = output * dropoutMask_;
      return output;
    }

  private:
    ParametersGRUFast params_;
    Expr dropoutMask_;
};

/***************************************************************/

template <class Cell = Tanh>
class RNN {
  public:

    template <class Parameters>
    RNN(const Parameters& params)
    : cell_(params) {}

    RNN(const Cell& cell)
    : cell_(cell) {}

    template <typename Expressions>
    std::vector<Expr> apply(const std::vector<Expressions>& inputs,
                            const Expr initialState) {
      return apply(inputs.begin(), inputs.end(),
                   initialState);
    }

    template <class Iterator>
    std::vector<Expr> apply(Iterator it, Iterator end,
                            const Expr initialState) {

      //auto xW = dot(input, params_.W);
      //
      //std::vector<Expr> outputs;
      //auto state = initialState;
      //for(int i = 0; i < input->shape()[2]; ++i) {
      //  auto x = view(xW, i, {dimBatch_, dimSrcEmb_});
      //  state = apply(cell_, *it++, state);
      //  outputs.push_back(state);
      //}
      //return concatenate(outputs, axis=2);

      std::vector<Expr> outputs;
      auto state = initialState;
      while(it != end) {
        state = apply(cell_, *it++, state);
        outputs.push_back(state);
      }
      return outputs;
    }

    Expr apply(Cell& cell, Expr input, Expr state) {
      return cell_.apply(input, state);
    }

    Expr apply(Cell& cell, std::pair<Expr, Expr> inputMask, Expr state) {
      return cell_.apply(inputMask.first, state, inputMask.second);
    }

    Cell& getCell() {
      return cell_;
    }

  private:
    Cell cell_;
};

/***************************************************************/

struct ParametersGRUWithAttention {
  // First GRU
  Expr U, W, b;

  // Attention
  Expr Wa, ba, Ua, va;

  // Conditional GRU
  Expr Uc, Wc, bc;
  float dropout = 0;
};

class GRUWithAttention {
  public:
    GRUWithAttention(ParametersGRUWithAttention params,
                     Expr context,
                     Expr softmaxMask = nullptr,
                     Expr dropoutMask = nullptr)
    : params_(params),
      context_(context),
      softmaxMask_(nullptr),
      dropoutMask_(dropoutMask) {
        mappedContext_ = dot(context_, params_.Ua);

        if(softmaxMask) {
          Shape shape = { softmaxMask->shape()[2],
                          softmaxMask->shape()[0] };
          softmaxMask_ = transpose(
            reshape(softmaxMask, shape));
        }
      }

    Expr apply(Expr input, Expr state, Expr mask = nullptr) {
      using namespace keywords;

    
      auto xW = dot(input, params_.W);
      auto sU = dot(state, params_.U);
      auto hidden = mask ?
        grufast({state, xW, sU, params_.b, mask}) :
        grufast({state, xW, sU, params_.b});

      int dimBatch = context_->shape()[0];
      int srcWords = context_->shape()[2];

      if(dropoutMask_)
        hidden = hidden * dropoutMask_;

      auto mappedState = dot(hidden, params_.Wa);
      auto temp = tanhPlus3(mappedState, mappedContext_ , params_.ba);

      // @TODO: horrible ->
      auto e = reshape(
        transpose(
          softmax(
            transpose(
              reshape(
                dot(temp, params_.va),
                {srcWords, dimBatch})),
            softmaxMask_)),
        {dimBatch, 1, srcWords});
      // <- horrible

      auto alignedSource = weighted_average(context_, e, axis=2);
      contexts_.push_back(alignedSource);

      auto aWc = dot(alignedSource, params_.Wc);
      auto hUc = dot(hidden, params_.Uc);
      auto output = mask ?
        grufast({hidden, aWc, hUc, params_.bc, mask}, true) :
        grufast({hidden, aWc, hUc, params_.bc}, true);

      if(dropoutMask_)
        output = output * dropoutMask_;

      return output;
    }

    std::vector<Expr>& getContexts() {
      return contexts_;
    }

  private:
    ParametersGRUWithAttention params_;
    Expr context_;
    Expr softmaxMask_;
    Expr dropoutMask_;
    Expr mappedContext_;
    std::vector<Expr> contexts_;
};


}
