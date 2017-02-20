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
#include "layers/attention.h"

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
        //std::reverse(states.begin(), states.end());
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
class BiRNN : public Layer {
  public:
    int dimState_;

    Ptr<RNN<Cell>> rnn1_;
    Ptr<RNN<Cell>> rnn2_;

    template <typename ...Args>
    BiRNN(Ptr<ExpressionGraph> graph,
          const std::string& name,
          int dimInput,
          int dimState,
          Args ...args)
    : Layer(name),
      dimState_{dimState},
      rnn1_(New<RNN<Cell>>(graph, name, dimInput, dimState,
                           keywords::direction=dir::forward,
                           args...)),
      rnn2_(New<RNN<Cell>>(graph, name + "_r", dimInput, dimState,
                           keywords::direction=dir::backward,
                           args...)) {}

    template <typename ...Args>
    Expr operator()(Expr input, Args ...args) {
      auto graph = input->graph();
      int dimBatch = input->shape()[0];
      auto startState = graph->zeros(keywords::shape={dimBatch, dimState_});
      return (*this)(input, startState, args...);
    }

    template <typename ...Args>
    Expr operator()(Expr input, Expr state, Args ...args) {
      Expr mask = Get(keywords::mask, nullptr, args...);

      auto graph = input->graph();
      int dimInput = input->shape()[1];

      auto states1 = rnn1_->apply(input, state, nullptr);
      auto states2 = rnn2_->apply(input, state, mask, true);

      std::reverse(states2.begin(), states2.end());
      std::vector<Expr> states;
      for(int i = 0; i < states1.size(); ++i)
        states.push_back(concatenate({states1[i], states2[i]},
                                     keywords::axis=1));

      return concatenate(states, keywords::axis=2);
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
    Expr U_, W_, b_;
    Expr gamma1_;
    Expr gamma2_;
    bool final_;
    bool layerNorm_;

  public:

    template <typename ...Args>
    GRU(ExpressionGraphPtr graph,
        const std::string prefix,
        int dimInput,
        int dimState,
        Args ...args) {

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

      if(layerNorm_) {
        gamma1_ = graph->param(prefix + "_gamma1", {1, 3 * dimState},
                               keywords::init=inits::from_value(1.f));
        gamma2_ = graph->param(prefix + "_gamma2", {1, 3 * dimState},
                               keywords::init=inits::from_value(1.f));
      }
    }

    Expr apply(Expr input, Expr state, Expr mask = nullptr) {
      return apply2(apply1(input), state, mask);
    }

    Expr apply1(Expr input) {
      auto xW = dot(input, W_);
      if(layerNorm_)
        xW = layer_norm(xW, gamma1_);
      return xW;
    }

    Expr apply2(Expr xW, Expr state, Expr mask = nullptr) {
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

    Expr getContexts() {
      return concatenate(att_->getContexts(), keywords::axis=2);
    }

    Expr getLastContext() {
      return att_->getContexts().back();
    }
};

typedef AttentionCell<GRU, GlobalAttention, GRU> CGRU;

}
