#pragma once

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <string>
#include <cstdio>
#include <boost/timer/timer.hpp>

#include "marian.h"
#include "graph/node_operators_binary.h"
#include "graph/expression_graph.h"

#include "layers/generic.h"

namespace marian {

class Tanh {
  private:
    Expr U_, W_, b_;

  public:
    template <typename ...Args>
    void initialize(
        ExpressionGraphPtr graph,
        const std::string prefix,
        int dimInput,
        int dimState,
        Args ...args) {
      U_ = graph->param(prefix + "_U", {dimState, dimState},
                        keywords::init=inits::glorot_uniform);
      W_ = graph->param(prefix + "_W", {dimInput, dimState},
                        keywords::init=inits::glorot_uniform);
      b_ = graph->param(prefix + "_b", {1, dimState},
                        keywords::init=inits::zeros);
    }

    Expr apply(Expr input, Expr state, Expr mask = nullptr) {
      return apply2(apply1(input), state, mask);
    }

    Expr apply1(Expr input) {
      return dot(input, W_);
    }

    Expr apply2(Expr xW, Expr state, Expr mask = nullptr) {
      auto output = tanh(xW, dot(state, U_), b_);
      if(mask)
        return output * mask;
      else
        return output;
    }
};

/***************************************************************/

template <class Cell>
class RNN : public Layer {
  public:
    int dimInput_;
    int dimState_;
    dir direction_;
    bool outputLast_;

    Ptr<Cell> cell_;

    template <typename ...Args>
    RNN(Ptr<ExpressionGraph> graph,
        const std::string& name,
        int dimInput,
        int dimState,
        Args ...args)
    : Layer(name),
      dimInput_{dimInput},
      dimState_{dimState},
      direction_{Get(keywords::direction, dir::forward, args...)},
      outputLast_{Get(keywords::output_last, false, args...)},
      cell_(New<Cell>(graph, name_, dimInput_, dimState_, args...)) {}

    Ptr<Cell> getCell() {
      return cell_;
    }

    std::vector<Expr> apply(const Expr input, const Expr initialState,
                            const Expr mask = nullptr, bool reverse = false) {
      auto xW = cell_->apply1(input);
      std::vector<Expr> outputs;
      auto state = initialState;
      for(size_t i = 0; i < input->shape()[2]; ++i) {
        int j = i;
        if(reverse)
          j = input->shape()[2] - i - 1;

        if(mask)
          state = cell_->apply2(step(xW, j), state, step(mask, j));
        else
          state = cell_->apply2(step(xW, j), state);
        outputs.push_back(state);
      }
      return outputs;
    }

    Expr apply(Ptr<Cell> cell, Expr input, Expr state) {
      return cell_->apply(input, state);
    }

    template <typename ...Args>
    Expr operator()(Expr input, Args ...args) {
      auto graph = input->graph();
      int dimBatch = input->shape()[0];
      auto startState = graph->zeros(keywords::shape={dimBatch, dimState_});
      return (*this)(input, startState, args...);
    }

    template <typename ...Args>
    Expr operator()(Expr input, Expr state, Args ...args) {
      auto graph = input->graph();
      int dimInput = input->shape()[1];

      Expr mask = Get(keywords::mask, nullptr, args...);

      if(direction_ == dir::backward) {
        auto states = apply(input, state, mask, true);
        std::reverse(states.begin(), states.end());
        if(outputLast_)
          return states.back();
        else
          return concatenate(states, keywords::axis=2);
      }
      else if(direction_ == dir::bidirect) {
        UTIL_THROW2("Use BiRNN for bidirectional RNNs");
      }
      else { // assuming dir::forward
        auto states = apply(input, state, mask, false);
        if(outputLast_)
          return states.back();
        else
          return concatenate(states, keywords::axis=2);
      }
    }
};

template <class Cell>
class MLRNN : public Layer {
  private:
    int layers_;
    bool skip_;
    bool skipFirst_;
    int dimState_;
    std::vector<Ptr<RNN<Cell>>> rnns_;

  public:

    template <typename ...Args>
    MLRNN(Ptr<ExpressionGraph> graph,
          const std::string& name,
          int layers,
          int dimInput,
          int dimState,
          Args ...args)
    : Layer(name),
      layers_(layers),
      skip_(Get(keywords::skip, false, args...)),
      skipFirst_(Get(keywords::skip_first, false, args...)),
      dimState_{dimState} {
      for(int i = 0; i < layers; ++i) {
        rnns_.push_back(
          New<RNN<Cell>>(graph,
                         name + "_l" + std::to_string(i),
                         i == 0 ? dimInput : dimState,
                         dimState,
                         args...)
        );
      }
    }

    template <typename ...Args>
    std::tuple<Expr, std::vector<Expr>>
    operator()(Expr input, Args ...args) {
      Expr output;
      std::vector<Expr> outStates;
      for(int i = 0; i < layers_; ++i) {
        auto outState = (*rnns_[i])(input, args...);
        outStates.push_back(outState);

        if(skip_ && (skipFirst_ || i > 0))
          output = outState + input;
        else
          output = outState;

        input = output;
      }
      return std::make_tuple(output, outStates);
    }

    template <typename ...Args>
    std::tuple<Expr, std::vector<Expr>>
    operator()(Expr input,
               std::vector<Expr> states,
               Args ...args) {
      Expr output;
      std::vector<Expr> outStates;
      for(int i = 0; i < layers_; ++i) {
        auto outState = (*rnns_[i])(input, states[i], args...);
        outStates.push_back(outState);

        if(skip_ && (skipFirst_ || i > 0))
          output = outState + input;
        else
          output = outState;

        input = output;
      }
      return std::make_tuple(output, outStates);
    }
};

template <class Cell>
class BiRNN : public Layer {
  public:
    int layers_;
    int dimState_;

    Ptr<RNN<Cell>> rnn1_;
    Ptr<RNN<Cell>> rnn2_;

    template <typename ...Args>
    BiRNN(Ptr<ExpressionGraph> graph,
          const std::string& name,
          int layers,
          int dimInput,
          int dimState,
          Args ...args)
    : Layer(name),
      dimState_{dimState},
      rnn1_(New<MLRNN<Cell>>(graph, name, layers, dimInput, dimState,
                             keywords::direction=dir::forward,
                             args...)),
      rnn2_(New<MLRNN<Cell>>(graph, name + "_r", layers, dimInput, dimState,
                             keywords::direction=dir::backward,
                             args...)) {}

    template <typename ...Args>
    std::vector<Expr> operator()(Expr input, Args ...args) {
      Expr mask = Get(keywords::mask, nullptr, args...);
      auto statesfw = (*rnn1_)(input);
      auto statesbw = (*rnn2_)(input, keywords::mask=mask);

      std::vector<Expr> outStates;
      for(int i = 0; i < layers_; ++i)
        outStates.push_back(concatenate({statesfw[i], statesbw[i]},
                                        keywords::axis=1));
      return outStates;
    }

    template <typename ...Args>
    std::vector<Expr> operator()(Expr input, std::vector<Expr> states, Args ...args) {
      Expr mask = Get(keywords::mask, nullptr, args...);
      auto statesfw = (*rnn1_)(input, states);
      auto statesbw = (*rnn2_)(input, states, keywords::mask=mask);

      std::vector<Expr> outStates;
      for(int i = 0; i < layers_; ++i)
        outStates.push_back(concatenate({statesfw[i], statesbw[i]},
                                        keywords::axis=1));
      return outStates;
    }
};

/***************************************************************/

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

/***************************************************************/

class GRU {
  private:
    std::string prefix_;

    Expr U_, W_, b_;
    Expr gamma1_;
    Expr gamma2_;

    bool final_;
    bool layerNorm_;
    float dropout_;

    Expr dropMaskX_;
    Expr dropMaskS_;

  public:

    template <typename ...Args>
    GRU(ExpressionGraphPtr graph,
        const std::string prefix,
        int dimInput,
        int dimState,
        Args ...args) : prefix_(prefix) {

      auto U = graph->param(prefix + "_U", {dimState, 2 * dimState},
                               keywords::init=inits::glorot_uniform);
      auto W = graph->param(prefix + "_W", {dimInput, 2 * dimState},
                               keywords::init=inits::glorot_uniform);
      auto b = graph->param(prefix + "_b", {1, 2 * dimState},
                               keywords::init=inits::zeros);
      auto Ux = graph->param(prefix + "_Ux", {dimState, dimState},
                                keywords::init=inits::glorot_uniform);
      auto Wx = graph->param(prefix + "_Wx", {dimInput, dimState},
                                keywords::init=inits::glorot_uniform);
      auto bx = graph->param(prefix + "_bx", {1, dimState},
                                keywords::init=inits::zeros);

      U_ = concatenate({U, Ux}, keywords::axis=1);
      W_ = concatenate({W, Wx}, keywords::axis=1);
      b_ = concatenate({b, bx}, keywords::axis=1);

      final_ = Get(keywords::final, false, args...);
      layerNorm_ = Get(keywords::normalize, false, args...);

      dropout_ = Get(keywords::dropout_prob, 0.0f, args...);

      if(layerNorm_) {
        gamma1_ = graph->param(prefix + "_gamma1", {1, 3 * dimState},
                               keywords::init=inits::from_value(1.f));
        gamma2_ = graph->param(prefix + "_gamma2", {1, 3 * dimState},
                               keywords::init=inits::from_value(1.f));
      }

      if(dropout_> 0.0f) {
        dropMaskX_ = graph->dropout(dropout_, {1, dimInput});
        dropMaskS_ = graph->dropout(dropout_, {1, dimState});
      }
    }

    Expr apply(Expr input, Expr state,
               Expr mask = nullptr) {
      return apply2(apply1(input), state, mask);
    }

    Expr apply1(Expr input) {
      if(dropMaskX_)
        input = dropout(input, keywords::mask=dropMaskX_);
      debug(input, "in");
      auto xW = dot(input, W_);
      if(layerNorm_)
        xW = layer_norm(xW, gamma1_);
      return xW;
    }

    Expr apply2(Expr xW, Expr state,
                Expr mask = nullptr) {
      if(dropMaskS_)
        state = dropout(state, keywords::mask=dropMaskS_);
      debug(state, "state");

      auto sU = dot(state, U_);

      if(layerNorm_)
        sU = layer_norm(sU, gamma2_);

      auto output = mask ?
        gruOps({state, xW, sU, b_, mask}, final_) :
        gruOps({state, xW, sU, b_}, final_);

      return output;
    }
};


/***************************************************************/

template <class Cell1, class Attention, class Cell2>
class AttentionCell {
  private:
    Ptr<Cell1> cell1_;
    Ptr<Cell2> cell2_;
    Ptr<Attention> att_;

  public:

    template <class ...Args>
    AttentionCell(Ptr<ExpressionGraph> graph,
                  const std::string prefix,
                  int dimInput,
                  int dimState,
                  Ptr<Attention> att,
                  Args ...args)
    {
      cell1_ = New<Cell1>(graph,
                          prefix + "_cell1",
                          dimInput,
                          dimState,
                          keywords::final=false,
                          args...);

      att_ = New<Attention>(att);

      cell2_ = New<Cell2>(graph,
                          prefix + "_cell2",
                          att_->outputDim(),
                          dimState,
                          keywords::final=true,
                          args...);
    }

    Expr apply(Expr input, Expr state, Expr mask = nullptr) {
      return apply2(apply1(input), state, mask);
    }

    Expr apply1(Expr input) {
      return cell1_->apply1(input);
    }

    Expr apply2(Expr xW, Expr state, Expr mask = nullptr) {
      auto hidden = cell1_->apply2(xW, state, mask);
      auto alignedSourceContext = att_->apply(hidden);
      return cell2_->apply(alignedSourceContext, hidden, mask);
    }

    Ptr<Attention> getAttention() {
      return att_;
    }

    Expr getContexts() {
      return concatenate(att_->getContexts(), keywords::axis=2);
    }

    Expr getLastContext() {
      return att_->getContexts().back();
    }
};

}
