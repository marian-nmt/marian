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

    if(cell) {
      return RNNState{
        reshape(rows(output, indices), {1, dimState, 1, numSelected}),
        reshape(rows(cell, indices), {1, dimState, 1, numSelected})
      };
    }
    else {
      return RNNState{
        reshape(rows(output, indices), {1, dimState, 1, numSelected}),
        nullptr
      };
    }
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

class Cell {
private:
  int dimInput_;
  int dimState_;

public:
  Cell(int dimInput, int dimState)
    : dimInput_(dimInput), dimState_(dimState) {}

  virtual int dimInput() {
    return dimInput_;
  }

  virtual int dimState() {
    return dimState_;
  }

  virtual RNNState apply(std::vector<Expr>, RNNState, Expr = nullptr) = 0;
  virtual std::vector<Expr> applyInput(std::vector<Expr> inputs) = 0;
  virtual RNNState applyState(std::vector<Expr>, RNNState, Expr = nullptr) = 0;
  virtual size_t numStates() = 0;
};

class RNNBase {
public:
  virtual Ptr<Cell> getCell() = 0;
  virtual RNNStates apply(Expr, RNNState, Expr, bool) = 0;
  virtual RNNStates operator()(Expr, Expr = nullptr) = 0;
  virtual RNNStates operator()(Expr, RNNState state, Expr = nullptr) = 0;
};


class RNN : public RNNBase {
private:
  Ptr<Cell> cell_;
  dir direction_;

public:

  template <typename... Args>
  RNN(Ptr<Cell> cell, Args... args)
      : cell_(cell),
        direction_{Get(keywords::direction, dir::forward, args...)}
      {}

  virtual Ptr<Cell> getCell() { return cell_; }

  virtual RNNStates apply(const Expr input,
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

  virtual RNNStates operator()(Expr input, Expr mask = nullptr) {
    auto graph = input->graph();
    int dimBatch = input->shape()[0];
    int dimState = cell_->dimState();

    auto output = graph->zeros(keywords::shape = {dimBatch, dimState});
    Expr cell = output;
    RNNState startState{ output, cell };

    return (*this)(input, startState, mask);
  }

  virtual RNNStates operator()(Expr input, RNNState state, Expr mask = nullptr) {
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

/*
template <class Cell>
class MLRNN : public Layer {
private:
  int layers_;
  bool skip_;
  bool skipFirst_;
  int dimState_;
  std::vector<Ptr<RNNBase>> rnns_;

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
*/

}
