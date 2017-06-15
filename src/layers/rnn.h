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

struct RNNState {
  Expr output;
  Expr cell;

  RNNState select(const std::vector<size_t>& indices) {

    int numSelected = indices.size();
    int dimState = output->shape()[1];

    if(cell)
      return RNNState{
        reshape(rows(output, indices), {1, dimState, 1, numSelected}),
        reshape(rows(cell, indices), {1, dimState, 1, numSelected})
      };
    else
      return RNNState{
        reshape(rows(output, indices), {1, dimState, 1, numSelected}),
        nullptr
      };
  }
};

class RNNStates {
  private:
    std::vector<RNNState> states_;

  public:

    auto begin() -> decltype(states_.begin()) { return states_.begin(); }
    auto end() -> decltype(states_.begin()) { return states_.end(); }

    Expr outputs() {
      std::vector<Expr> outputs;
      for(auto s : states_)
        outputs.push_back(s.output);
      if(outputs.size() > 1)
        return concatenate(outputs, keywords::axis = 2);
      else
        return outputs[0];
    }

    RNNState& operator[](size_t i) { return states_[i]; };
    const RNNState& operator[](size_t i) const { return states_[i]; };

    size_t size() const { return states_.size(); };

    void push_back(const RNNState& state) {
      states_.push_back(state);
    }

    RNNStates select(const std::vector<size_t>& indices) {
      RNNStates selected;
      for(auto& state : states_)
        selected.push_back(state.select(indices));
      return selected;
    }

    void reverse() {
      std::reverse(states_.begin(), states_.end());
    }
};

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

  RNNStates apply(const Expr input,
                  const RNNState initialState,
                  const Expr mask = nullptr,
                  bool reverse = false) {

    RNNState state = initialState;

    auto xWs = cell_->applyInput({input});

    size_t timeSteps = input->shape()[2];

    RNNStates outputs;
    for(size_t i = 0; i < timeSteps; ++i) {
      int j = i;
      if(reverse)
        j = timeSteps - i - 1;

      std::vector<Expr> steps(xWs.size());
      std::transform(xWs.begin(), xWs.end(), steps.begin(),
                     [j](Expr e) { return step(e, j); });

      if(mask)
        state = cell_->applyState(steps, state, step(mask, j));
      else
        state = cell_->applyState(steps, state);

      outputs.push_back(state);
    }
    return outputs;
  }

  template <typename... Args>
  RNNStates operator()(Expr input, Args... args) {
    auto graph = input->graph();
    int dimBatch = input->shape()[0];

    auto output = graph->zeros(keywords::shape = {dimBatch, dimState_});
    Expr cell = output;

    // TODO: look at this again
    RNNState startState{ output, cell };

    return (*this)(input, startState, args...);
  }

  template <typename... Args>
  RNNStates operator()(Expr input, RNNState state, Args... args) {
    auto graph = input->graph();
    int dimInput = input->shape()[1];

    Expr mask = Get(keywords::mask, nullptr, args...);

    if(direction_ == dir::backward) {
      auto states = apply(input, state, mask, true);
      states.reverse();
      return states;
    } else {  // assuming dir::forward
      auto states = apply(input, state, mask, false);
      return states;
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

  static size_t numStates() { return 1; }
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

  RNNState apply(std::vector<Expr> inputs,
                 RNNState state,
                 Expr mask = nullptr) {
    return applyState(applyInput(inputs), state, mask);
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

  RNNState applyState(std::vector<Expr> xWs,
                      RNNState state,
                      Expr mask = nullptr) {

    auto stateOrig = state.output;
    auto stateDropped = stateOrig;
    if(dropMaskS_)
      stateDropped = dropout(stateOrig, keywords::mask = dropMaskS_);

    auto sU = dot(stateDropped, U_);

    if(layerNorm_)
      sU = layer_norm(sU, gamma2_);

    auto xW = xWs.front();

    auto output = mask ? gruOps({stateOrig, xW, sU, b_, mask}, final_) :
                         gruOps({stateOrig, xW, sU, b_}, final_);

    return {output, nullptr}; // no cell state, hence nullptr
  }

  static size_t numStates() { return 1; }
};

/***************************************************************/

class SlowLSTM {
private:
  std::string prefix_;

  Expr Uf_, Wf_, bf_;
  Expr Ui_, Wi_, bi_;
  Expr Uo_, Wo_, bo_;
  Expr Uc_, Wc_, bc_;

public:
  template <typename... Args>
  SlowLSTM(Ptr<ExpressionGraph> graph,
      const std::string prefix,
      int dimInput,
      int dimState,
      Args... args)
      : prefix_(prefix) {

    Uf_ = graph->param(prefix + "_Uf", {dimState, dimState},
                       keywords::init=inits::glorot_uniform);
    Wf_ = graph->param(prefix + "_Wf", {dimInput, dimState},
                       keywords::init=inits::glorot_uniform);
    bf_ = graph->param(prefix + "_bf", {1, dimState},
                       keywords::init=inits::zeros);

    Ui_ = graph->param(prefix + "_Ui", {dimState, dimState},
                       keywords::init=inits::glorot_uniform);
    Wi_ = graph->param(prefix + "_Wi", {dimInput, dimState},
                       keywords::init=inits::glorot_uniform);
    bi_ = graph->param(prefix + "_bi", {1, dimState},
                       keywords::init=inits::zeros);

    Uo_ = graph->param(prefix + "_Uo", {dimState, dimState},
                       keywords::init=inits::glorot_uniform);
    Wo_ = graph->param(prefix + "_Wo", {dimInput, dimState},
                       keywords::init=inits::glorot_uniform);
    bo_ = graph->param(prefix + "_bo", {1, dimState},
                       keywords::init=inits::zeros);

    Uc_ = graph->param(prefix + "_Uc", {dimState, dimState},
                       keywords::init=inits::glorot_uniform);
    Wc_ = graph->param(prefix + "_Wc", {dimInput, dimState},
                       keywords::init=inits::glorot_uniform);
    bc_ = graph->param(prefix + "_bc", {1, dimState},
                       keywords::init=inits::zeros);

  }

  RNNState apply(std::vector<Expr> inputs,
                 RNNState state,
                 Expr mask = nullptr) {
    return applyState(applyInput(inputs), state, mask);
  }

  std::vector<Expr> applyInput(std::vector<Expr> inputs) {
    Expr input;
    if(inputs.size() > 1)
      input = concatenate(inputs, keywords::axis = 1);
    else
      input = inputs.front();

    auto xWf = dot(input, Wf_);
    auto xWi = dot(input, Wi_);
    auto xWo = dot(input, Wo_);
    auto xWc = dot(input, Wc_);

    return {xWf, xWi, xWo, xWc};
  }

  RNNState applyState(std::vector<Expr> xWs,
                      RNNState state,
                      Expr mask = nullptr) {
    auto recState = state.output;
    auto cellState = state.cell;

    auto sUf = affine(recState, Uf_, bf_);
    auto sUi = affine(recState, Ui_, bi_);
    auto sUo = affine(recState, Uo_, bo_);
    auto sUc = affine(recState, Uc_, bc_);

    auto f = logit(xWs[0] + sUf);
    auto i = logit(xWs[1] + sUi);
    auto o = logit(xWs[2] + sUo);
    auto c = tanh(xWs[3] + sUc);

    auto nextCellState = f * cellState + i * c;
    auto maskedCellState = mask ? mask * nextCellState : nextCellState;

    auto nextState = o * tanh(maskedCellState);
    auto maskedState = mask ? mask * nextState : nextState;

    return {maskedState, maskedCellState};
  }

  static size_t numStates() { return 2; }
};

/***************************************************************/

Expr lstmOpsC(const std::vector<Expr>& nodes);
Expr lstmOpsO(const std::vector<Expr>& nodes);

class FastLSTM {
private:
  std::string prefix_;

  Expr U_, W_, b_;
  Expr gamma1_;
  Expr gamma2_;

  bool layerNorm_;
  float dropout_;

  Expr dropMaskX_;
  Expr dropMaskS_;
//  Expr dropMaskC_;

public:
  template <typename... Args>
  FastLSTM(Ptr<ExpressionGraph> graph,
      const std::string prefix,
      int dimInput,
      int dimState,
      Args... args)
      : prefix_(prefix) {

    U_ = graph->param(prefix + "_U", {dimState, 4 * dimState},
                      keywords::init=inits::glorot_uniform);
    W_ = graph->param(prefix + "_W", {dimInput, 4 * dimState},
                      keywords::init=inits::glorot_uniform);
    b_ = graph->param(prefix + "_b", {1, 4 * dimState},
                      keywords::init=inits::zeros);

    layerNorm_ = Get(keywords::normalize, false, args...);

    dropout_ = Get(keywords::dropout_prob, 0.0f, args...);
    if(dropout_ > 0.0f) {
      dropMaskX_ = graph->dropout(dropout_, {1, dimInput});
      dropMaskS_ = graph->dropout(dropout_, {1, dimState});
    }

    if(layerNorm_) {
      gamma1_ = graph->param(prefix + "_gamma1",
                             {1, 4 * dimState},
                             keywords::init = inits::from_value(1.f));
      gamma2_ = graph->param(prefix + "_gamma2",
                             {1, 4 * dimState},
                             keywords::init = inits::from_value(1.f));
    }
  }

  RNNState apply(std::vector<Expr> inputs,
                 RNNState state,
                 Expr mask = nullptr) {
    return applyState(applyInput(inputs), state, mask);
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

  RNNState applyState(std::vector<Expr> xWs,
                      RNNState state,
                      Expr mask = nullptr) {

    auto recState = state.output;
    auto cellState = state.cell;

    auto recStateDropped = recState;
    if(dropMaskS_)
      recStateDropped = dropout(recState, keywords::mask = dropMaskS_);

    auto sU = dot(recStateDropped, U_);

    if(layerNorm_)
      sU = layer_norm(sU, gamma2_);

    auto xW = xWs.front();

    // dc/dp where p = W_i, U_i, ..., but without index o
    auto nextCellState = mask ?
      lstmOpsC({cellState, xW, sU, b_, mask}) :
      lstmOpsC({cellState, xW, sU, b_});

    // dh/dp dh/dc where p = W_o, U_o, b_o
    auto nextRecState = mask ?
      lstmOpsO({nextCellState, xW, sU, b_, mask}) :
      lstmOpsO({nextCellState, xW, sU, b_});

    return {nextRecState, nextCellState};
  }

  static size_t numStates() { return 2; }
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

  RNNState apply(std::vector<Expr> inputs,
                 RNNState state,
                 Expr mask = nullptr) {
    return applyState(applyInput(inputs), state, mask);
  }

  std::vector<Expr> applyInput(std::vector<Expr> inputs) {
    return cell1_->applyInput(inputs);
  }

  RNNState applyState(std::vector<Expr> xWs,
                      RNNState state,
                      Expr mask = nullptr) {
    if(Cell1::numStates == Cell2::numStates) {
      auto hidden = cell1_->applyState(xWs, state, mask);
      auto alignedSourceContext = att_->apply(hidden.output);
      //debug(alignedSourceContext, "aligned");
      return cell2_->apply({alignedSourceContext}, hidden, mask);
    }
    else if(Cell1::numStates > Cell2::numStates) {
      auto hidden = cell1_->applyState(xWs, state, mask);
      auto alignedSourceContext = att_->apply(hidden.output);
      auto output = cell2_->apply({alignedSourceContext}, hidden, mask);
      return { output.output, hidden.cell };
    }
    else {
      auto hidden = cell1_->applyState(xWs, state, mask);
      auto alignedSourceContext = att_->apply(hidden.output);
      return cell2_->apply({alignedSourceContext}, {hidden.output, state.cell}, mask);
    }
  }

  Ptr<Attention> getAttention() { return att_; }

  Expr getContexts() {
    return concatenate(att_->getContexts(), keywords::axis = 2);
  }

  Expr getLastContext() { return att_->getContexts().back(); }

  static size_t numStates() { return Cell1::numStates(); }
};

typedef AttentionCell<GRU, GlobalAttention, GRU> CGRU;

typedef FastLSTM LSTM;

typedef AttentionCell<LSTM, GlobalAttention, LSTM> CLSTM;
typedef AttentionCell<LSTM, GlobalAttention, GRU> CLSTMGRU;
}
