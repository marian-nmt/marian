#include <algorithm>
#include <chrono>
#include <iomanip>
#include <string>
#include <cstdio>
#include <boost/timer/timer.hpp>

#include "marian.h"
#include "graph/node_operators_binary.h"
#include "graph/expression_graph.h"

#include "layers/attention.h"

namespace marian {

struct ParametersTanh {
  Expr U, W, b;
  float dropout = 0;
};

class Tanh {
  public:
    Tanh(ParametersTanh params)
    : params_(params) {}

    Expr apply(Expr input, Expr state, Expr mask = nullptr) {
      return apply2(apply1(input), state, mask);
    }

    Expr apply1(Expr input) {
      return dot(input, params_.W);
    }

    Expr apply2(Expr xW, Expr state, Expr mask = nullptr) {
      using namespace keywords;

      Expr output = xW + dot(state, params_.U);
      if(params_.b)
        output = output + params_.b;
      output = tanh(output);

      if(mask)
        return output * mask;
      else
        return output;
    }

  private:
    ParametersTanh params_;
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
      std::vector<Expr> outputs;
      auto state = initialState;
      while(it != end) {
        state = apply(cell_, *it++, state);
        outputs.push_back(state);
      }
      return outputs;
    }

    std::vector<Expr> apply(const Expr input, const Expr initialState,
                            const Expr mask = nullptr, bool reverse = false) {
      auto xW = cell_.apply1(input);

      std::vector<Expr> outputs;
      auto state = initialState;
      for(size_t i = 0; i < input->shape()[2]; ++i) {
        int j = i;
        if(reverse)
          j = input->shape()[2] - i - 1;

        if(mask)
          state = cell_.apply2(step(xW, j), state, step(mask, j));
        else
          state = cell_.apply2(step(xW, j), state);
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

struct ParametersGRU {
  Expr U, W, b;
  bool final{false};
};

struct GRUFastNodeOp : public NaryNodeOp {
  bool final_;

  template <typename ...Args>
  GRUFastNodeOp(const std::vector<Expr>& nodes, bool final, Args ...args)
    : NaryNodeOp(nodes,
                 args...),
      final_(final) {}

  NodeOps forwardOps() {
    std::vector<Tensor> inputs;
    for(auto child : children_)
      inputs.push_back(child->val());

    return {
      NodeOp(GRUFastForward(val_, inputs, final_))
    };
  }

  NodeOps backwardOps() {
    std::vector<Tensor> inputs;
    std::vector<Tensor> outputs;
    for(auto child : children_) {
      inputs.push_back(child->val());
      if(child->trainable())
        outputs.push_back(child->grad());
      else
        outputs.push_back(nullptr);
    }

    return {
      NodeOp(GRUFastBackward(outputs, inputs, adj_, final_))
    };
  }

  // do not check if node is trainable
  virtual void runBackward(const NodeOps& ops) {
    for(auto&& op : ops)
      op();
  }

  const std::string type() {
    return "GRU-ops";
  }

  const std::string color() {
    return "yellow";
  }
};

Expr gruOps(const std::vector<Expr>& nodes, bool final = false) {
  return Expression<GRUFastNodeOp>(nodes, final);
}

class GRU {
  public:
    GRU(ParametersGRU params)
    : params_(params) {}

    Expr apply(Expr input, Expr state, Expr mask = nullptr) {
      return apply2(apply1(input), state, mask);
    }

    Expr apply1(Expr input) {
      auto xW = dot(input, params_.W);
      return xW;
    }

    Expr apply2(Expr xW, Expr state, Expr mask = nullptr) {
      auto sU = dot(state, params_.U);

      auto output = mask ?
        gruOps({state, xW, sU, params_.b, mask}, params_.final) :
        gruOps({state, xW, sU, params_.b}, params_.final);

      return output;
    }

  private:
    ParametersGRU params_;
};

/***************************************************************/

class ConditionalGRU {
  private:
    GRU gru1_;
    GRU gru2_;
    Attention att_;

  public:
    ConditionalGRU(ParametersGRU gruParams1,
                   ParametersGRU gruParams2,
                   Attention att)
     : gru1_(gruParams1),
       gru2_(gruParams2),
       att_(att)
    {}

    Expr apply(Expr input, Expr state, Expr mask = nullptr) {
      return apply2(apply1(input), state, mask);
    }

    Expr apply1(Expr input) {
      return gru1_.apply1(input);
    }

    Expr apply2(Expr xW, Expr state, Expr mask = nullptr) {
      auto hidden = gru1_.apply2(xW, state, mask);
      auto alignedSourceContext = att_.apply(hidden);
      return gru2_.apply(alignedSourceContext, hidden, mask);
    }

    Attention& getAttention() {
      return att_;
    }
};

}
