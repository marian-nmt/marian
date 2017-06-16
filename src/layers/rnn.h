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
    RNNStates() {}
    RNNStates(const std::vector<RNNState>& states) : states_(states) {}

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

    RNNState& back() { return states_.back(); }
    const RNNState& back() const { return states_.back(); }

    RNNState& front() { return states_.front(); }
    const RNNState& front() const { return states_.front(); }

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

    void clear() { states_.clear(); }
};

class Cell {
protected:
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
  virtual Expr operator()(Expr, Expr = nullptr) = 0;
  virtual Expr operator()(Expr, RNNState, Expr = nullptr) = 0;
  virtual Expr operator()(Expr, RNNStates, Expr = nullptr) = 0;
  virtual RNNStates last() = 0;
};


class RNN : public RNNBase {
private:
  Ptr<Cell> cell_;
  dir direction_;
  RNNStates last_;

  RNNStates apply(const Expr input,
                          const RNNStates initialState,
                          const Expr mask = nullptr) {
    last_.clear();

    RNNState state = initialState.front();

    auto xWs = cell_->applyInput({input});

    size_t timeSteps = input->shape()[2];

    RNNStates outputs;
    for(size_t i = 0; i < timeSteps; ++i) {
      int j = i;
      if(direction_ == dir::backward)
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

    if(direction_ == dir::backward)
      outputs.reverse();

    last_.push_back(outputs.back());

    return outputs;
  }

  RNNStates apply(const Expr input, const Expr mask = nullptr) {
    auto graph = input->graph();
    int dimBatch = input->shape()[0];
    int dimState = cell_->dimState();

    auto output = graph->zeros(keywords::shape = {dimBatch, dimState});
    Expr cell = output;
    RNNState startState{ output, cell };

    return apply(input, RNNStates({startState}), mask);
  }

public:

  template <typename... Args>
  RNN(Ptr<Cell> cell, Args... args)
      : cell_(cell),
        direction_{Get(keywords::direction, dir::forward, args...)}
      {}

  virtual Expr operator()(Expr input, Expr mask = nullptr) {
    return apply(input, mask).outputs();
  }

  virtual Expr operator()(Expr input, RNNStates states, Expr mask = nullptr) {
    return apply(input, states, mask).outputs();
  }

  virtual Expr operator()(Expr input, RNNState state, Expr mask = nullptr) {
    return apply(input, RNNStates({state}), mask).outputs();
  }

  RNNStates last() {
    return last_;
  }
};


class MLRNN : public RNNBase {
private:
  bool skip_;
  bool skipFirst_;
  std::vector<Ptr<RNNBase>> rnns_;

public:
  template <typename... Args>
  MLRNN(const std::vector<Ptr<Cell>>& cells,
        Args... args)
      : skip_(Get(keywords::skip, false, args...)),
        skipFirst_(Get(keywords::skip_first, false, args...)) {

    for(auto cell : cells)
      rnns_.push_back(New<RNN>(cell));
  }

  Expr operator()(Expr input, Expr mask = nullptr) {
    Expr output;
    for(int i = 0; i < rnns_.size(); ++i) {
      auto layerOutput = (*rnns_[i])(input, mask);

      if(skip_ && (skipFirst_ || i > 0))
        output = layerOutput + input;
      else
        output = layerOutput;

      input = output;
    }
    return output;
  }

  Expr operator()(Expr input, RNNStates states, Expr mask = nullptr) {
    Expr output;
    for(int i = 0; i < rnns_.size(); ++i) {
      auto layerOutput = (*rnns_[i])(input, RNNStates({states[i]}), mask);

      if(skip_ && (skipFirst_ || i > 0))
        output = layerOutput + input;
      else
        output = layerOutput;

      input = output;
    }
    return output;
  }

  Expr operator()(Expr input, RNNState state, Expr mask = nullptr) {
    Expr output;
    for(int i = 0; i < rnns_.size(); ++i) {
      auto layerOutput = (*rnns_[i])(input, RNNStates({state}), mask);

      if(skip_ && (skipFirst_ || i > 0))
        output = layerOutput + input;
      else
        output = layerOutput;

      input = output;
    }
    return output;
  }

  RNNStates last() {
    RNNStates temp;
    for(auto rnn : rnns_)
      temp.push_back(rnn->last().back());
    return temp;
  }

};

}
