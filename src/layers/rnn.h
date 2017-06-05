#pragma once

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <iomanip>
#include <string>

#include "common/definitions.h"
#include "graph/expression_graph.h"
#include "graph/expression_operators.h"
#include "layers/attention.h"
#include "layers/generic.h"

namespace marian {

template <class Cell>
class RNN : public Layer {
public:
  int dimInput_;
  int dimState_;
  dir direction_;
  bool outputLast_;

  Ptr<Cell> cell_;

  template <typename... Args>
  RNN(Ptr<ExpressionGraph> graph,
      const std::string& name,
      int dimInput,
      int dimState,
      Args... args)
      : Layer(name),
        dimInput_{dimInput},
        dimState_{dimState},
        direction_{Get(keywords::direction, dir::forward, args...)},
        outputLast_{Get(keywords::output_last, false, args...)},
        cell_(New<Cell>(graph, name_, dimInput_, dimState_, args...)) {}

  Ptr<Cell> getCell() { return cell_; }

  std::vector<Expr> apply(const Expr input,
                          const Expr initialState,
                          const Expr mask = nullptr,
                          bool reverse = false) {
    auto xWs = cell_->applyInput({input});

    std::vector<Expr> outputs;
    std::vector<Expr> states = {initialState};
    for(size_t i = 0; i < input->shape()[2]; ++i) {
      int j = i;
      if(reverse)
        j = input->shape()[2] - i - 1;

      std::vector<Expr> steps(xWs.size());
      std::transform(xWs.begin(), xWs.end(), steps.begin(), [j](Expr e) {
        return step(e, j);
      });

      if(mask)
        states = cell_->applyState(steps, states, step(mask, j));
      else
        states = cell_->applyState(steps, states);
      outputs.push_back(states.front());
    }
    return outputs;
  }

  Expr apply(Ptr<Cell> cell, Expr input, Expr state) {
    return cell_->apply(input, state).front();
  }

  template <typename... Args>
  Expr operator()(Expr input, Args... args) {
    auto graph = input->graph();
    int dimBatch = input->shape()[0];
    auto startState = graph->zeros(keywords::shape = {dimBatch, dimState_});
    return (*this)(input, startState, args...);
  }

  template <typename... Args>
  Expr operator()(Expr input, Expr state, Args... args) {
    auto graph = input->graph();
    int dimInput = input->shape()[1];

    Expr mask = Get(keywords::mask, nullptr, args...);

    if(direction_ == dir::backward) {
      auto states = apply(input, state, mask, true);
      std::reverse(states.begin(), states.end());
      if(outputLast_)
        return states.back();
      else
        return concatenate(states, keywords::axis = 2);
    } else {  // assuming dir::forward
      auto states = apply(input, state, mask, false);
      if(outputLast_)
        return states.back();
      else
        return concatenate(states, keywords::axis = 2);
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
  template <typename... Args>
  MLRNN(Ptr<ExpressionGraph> graph,
        const std::string& name,
        int layers,
        int dimInput,
        int dimState,
        Args... args)
      : Layer(name),
        layers_(layers),
        skip_(Get(keywords::skip, false, args...)),
        skipFirst_(Get(keywords::skip_first, false, args...)),
        dimState_{dimState} {
    for(int i = 0; i < layers; ++i) {
      rnns_.push_back(New<RNN<Cell>>(graph,
                                     name + "_l" + std::to_string(i),
                                     i == 0 ? dimInput : dimState,
                                     dimState,
                                     args...));
    }
  }

  template <typename... Args>
  std::tuple<Expr, std::vector<Expr>> operator()(Expr input, Args... args) {
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

  template <typename... Args>
  std::tuple<Expr, std::vector<Expr>> operator()(Expr input,
                                                 std::vector<Expr> states,
                                                 Args... args) {
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

/***************************************************************/

class Tanh {
private:
  Expr U_, W_, b_;
  Expr gamma1_;
  Expr gamma2_;

  bool layerNorm_;
  float dropout_;

  Expr dropMaskX_;
  Expr dropMaskS_;

public:
  template <typename... Args>
  Tanh(Ptr<ExpressionGraph> graph,
       const std::string prefix,
       int dimInput,
       int dimState,
       Args... args) {
    U_ = graph->param(prefix + "_U",
                      {dimState, dimState},
                      keywords::init = inits::glorot_uniform);
    W_ = graph->param(prefix + "_W",
                      {dimInput, dimState},
                      keywords::init = inits::glorot_uniform);
    b_ = graph->param(
        prefix + "_b", {1, dimState}, keywords::init = inits::zeros);

    layerNorm_ = Get(keywords::normalize, false, args...);

    dropout_ = Get(keywords::dropout_prob, 0.0f, args...);
    if(dropout_ > 0.0f) {
      dropMaskX_ = graph->dropout(dropout_, {1, dimInput});
      dropMaskS_ = graph->dropout(dropout_, {1, dimState});
    }

    if(layerNorm_) {
      gamma1_ = graph->param(prefix + "_gamma1",
                             {1, 3 * dimState},
                             keywords::init = inits::from_value(1.f));
      gamma2_ = graph->param(prefix + "_gamma2",
                             {1, 3 * dimState},
                             keywords::init = inits::from_value(1.f));
    }
  }

  std::vector<Expr> apply(std::vector<Expr> inputs,
                          std::vector<Expr> states,
                          Expr mask = nullptr) {
    return applyState(applyInput(inputs), states, mask);
  }

  std::vector<Expr> applyInput(std::vector<Expr> inputs) {
    Expr input;
    if(inputs.size() > 1)
      input = concatenate(inputs, keywords::axis = 1);
    else
      input = inputs.front();

    if(dropMaskX_)
      input = dropout(input, keywords::mask = dropMaskX_);

    auto xW = dot(input, W_);

    if(layerNorm_)
      xW = layer_norm(xW, gamma1_);

    return {xW};
  }

  std::vector<Expr> applyState(std::vector<Expr> xWs,
                               std::vector<Expr> states,
                               Expr mask = nullptr) {
    Expr state;
    if(states.size() > 1)
      state = concatenate(states, keywords::axis = 1);
    else
      state = states.front();

    auto stateDropped = state;
    if(dropMaskS_)
      stateDropped = dropout(state, keywords::mask = dropMaskS_);
    auto sU = dot(stateDropped, U_);
    if(layerNorm_)
      sU = layer_norm(sU, gamma2_);

    auto xW = xWs.front();

    auto output = tanh(xW, sU, b_);
    if(mask)
      return {output * mask};
    else
      return {output};
  }
};

Expr gruOps(const std::vector<Expr>& nodes, bool final = false);

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
  template <typename... Args>
  GRU(Ptr<ExpressionGraph> graph,
      const std::string prefix,
      int dimInput,
      int dimState,
      Args... args)
      : prefix_(prefix) {
    auto U = graph->param(prefix + "_U",
                          {dimState, 2 * dimState},
                          keywords::init = inits::glorot_uniform);
    auto W = graph->param(prefix + "_W",
                          {dimInput, 2 * dimState},
                          keywords::init = inits::glorot_uniform);
    auto b = graph->param(
        prefix + "_b", {1, 2 * dimState}, keywords::init = inits::zeros);
    auto Ux = graph->param(prefix + "_Ux",
                           {dimState, dimState},
                           keywords::init = inits::glorot_uniform);
    auto Wx = graph->param(prefix + "_Wx",
                           {dimInput, dimState},
                           keywords::init = inits::glorot_uniform);
    auto bx = graph->param(
        prefix + "_bx", {1, dimState}, keywords::init = inits::zeros);

    U_ = concatenate({U, Ux}, keywords::axis = 1);
    W_ = concatenate({W, Wx}, keywords::axis = 1);
    b_ = concatenate({b, bx}, keywords::axis = 1);

    // @TODO use this and adjust Amun model type saving and loading
    // U_ = graph->param(prefix + "_U", {dimState, 3 * dimState},
    //                  keywords::init=inits::glorot_uniform);
    // W_ = graph->param(prefix + "_W", {dimInput, 3 * dimState},
    //                  keywords::init=inits::glorot_uniform);
    // b_ = graph->param(prefix + "_b", {1, 3 * dimState},
    //                  keywords::init=inits::zeros);

    final_ = Get(keywords::final, false, args...);
    layerNorm_ = Get(keywords::normalize, false, args...);

    dropout_ = Get(keywords::dropout_prob, 0.0f, args...);
    if(dropout_ > 0.0f) {
      dropMaskX_ = graph->dropout(dropout_, {1, dimInput});
      dropMaskS_ = graph->dropout(dropout_, {1, dimState});
    }

    if(layerNorm_) {
      gamma1_ = graph->param(prefix + "_gamma1",
                             {1, 3 * dimState},
                             keywords::init = inits::from_value(1.f));
      gamma2_ = graph->param(prefix + "_gamma2",
                             {1, 3 * dimState},
                             keywords::init = inits::from_value(1.f));
    }
  }

  std::vector<Expr> apply(std::vector<Expr> inputs,
                          std::vector<Expr> states,
                          Expr mask = nullptr) {
    return applyState(applyInput(inputs), states, mask);
  }

  std::vector<Expr> applyInput(std::vector<Expr> inputs) {
    Expr input;
    if(inputs.size() > 1)
      input = concatenate(inputs, keywords::axis = 1);
    else
      input = inputs.front();

    if(dropMaskX_)
      input = dropout(input, keywords::mask = dropMaskX_);

    auto xW = dot(input, W_);

    if(layerNorm_)
      xW = layer_norm(xW, gamma1_);
    return {xW};
  }

  std::vector<Expr> applyState(std::vector<Expr> xWs,
                               std::vector<Expr> states,
                               Expr mask = nullptr) {
    Expr state;
    if(states.size() > 1)
      state = concatenate(states, keywords::axis = 1);
    else
      state = states.front();

    auto stateDropped = state;
    if(dropMaskS_)
      stateDropped = dropout(state, keywords::mask = dropMaskS_);

    auto sU = dot(stateDropped, U_);

    if(layerNorm_)
      sU = layer_norm(sU, gamma2_);

    auto xW = xWs.front();

    auto output = mask ? gruOps({state, xW, sU, b_, mask}, final_) :
                         gruOps({state, xW, sU, b_}, final_);

    return {output};
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
  template <class... Args>
  AttentionCell(Ptr<ExpressionGraph> graph,
                const std::string prefix,
                int dimInput,
                int dimState,
                Ptr<Attention> att,
                Args... args) {
    cell1_ = New<Cell1>(graph,
                        prefix + "_cell1",
                        dimInput,
                        dimState,
                        keywords::final = false,
                        args...);

    att_ = New<Attention>(att);

    cell2_ = New<Cell2>(graph,
                        prefix + "_cell2",
                        att_->outputDim(),
                        dimState,
                        keywords::final = true,
                        args...);
  }

  std::vector<Expr> apply(std::vector<Expr> inputs,
                          std::vector<Expr> states,
                          Expr mask = nullptr) {
    return applyState(applyInput(inputs), states, mask);
  }

  std::vector<Expr> applyInput(std::vector<Expr> inputs) {
    return cell1_->applyInput(inputs);
  }

  std::vector<Expr> applyState(std::vector<Expr> xWs,
                               std::vector<Expr> states,
                               Expr mask = nullptr) {
    auto hidden = cell1_->applyState(xWs, states, mask);
    auto alignedSourceContext = att_->apply(hidden.front());
    return cell2_->apply({alignedSourceContext}, hidden, mask);
  }

  Ptr<Attention> getAttention() { return att_; }

  Expr getContexts() {
    return concatenate(att_->getContexts(), keywords::axis = 2);
  }

  Expr getLastContext() { return att_->getContexts().back(); }
};

typedef AttentionCell<GRU, GlobalAttention, GRU> CGRU;
}
