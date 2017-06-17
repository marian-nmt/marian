#pragma once

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <iomanip>
#include <string>

#include "common/definitions.h"
#include "graph/expression_graph.h"
#include "graph/expression_operators.h"
#include "layers/generic.h"

#include "rnn/types.h"
#include "rnn/attention.h"

namespace marian {
namespace rnn {

class Base {
public:
  virtual Expr operator()(Expr, Expr = nullptr) = 0;
  virtual Expr operator()(Expr, State, Expr = nullptr) = 0;
  virtual Expr operator()(Expr, States, Expr = nullptr) = 0;
  virtual States last() = 0;
};


class RNN : public Base {
private:
  Ptr<Cell> cell_;
  dir direction_;
  States last_;

  States apply(const Expr input,
                          const States initialState,
                          const Expr mask = nullptr) {
    last_.clear();

    State state = initialState.front();

    auto xWs = cell_->applyInput({input});

    size_t timeSteps = input->shape()[2];

    States outputs;
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

  States apply(const Expr input, const Expr mask = nullptr) {
    auto graph = input->graph();
    int dimBatch = input->shape()[0];
    int dimState = cell_->getOptions()->get<int>("dimState");

    auto output = graph->zeros(keywords::shape = {dimBatch, dimState});
    Expr cell = output;
    State startState{ output, cell };

    return apply(input, States({startState}), mask);
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

  virtual Expr operator()(Expr input, States states, Expr mask = nullptr) {
    return apply(input, states, mask).outputs();
  }

  virtual Expr operator()(Expr input, State state, Expr mask = nullptr) {
    return apply(input, States({state}), mask).outputs();
  }

  States last() {
    return last_;
  }
};


class MLRNN : public Base {
private:
  bool skip_;
  bool skipFirst_;
  std::vector<Ptr<Base>> rnns_;

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

  Expr operator()(Expr input, States states, Expr mask = nullptr) {
    Expr output;
    for(int i = 0; i < rnns_.size(); ++i) {
      auto layerOutput = (*rnns_[i])(input, States({states[i]}), mask);

      if(skip_ && (skipFirst_ || i > 0))
        output = layerOutput + input;
      else
        output = layerOutput;

      input = output;
    }
    return output;
  }

  Expr operator()(Expr input, State state, Expr mask = nullptr) {
    Expr output;
    for(int i = 0; i < rnns_.size(); ++i) {
      auto layerOutput = (*rnns_[i])(input, States({state}), mask);

      if(skip_ && (skipFirst_ || i > 0))
        output = layerOutput + input;
      else
        output = layerOutput;

      input = output;
    }
    return output;
  }

  States last() {
    States temp;
    for(auto rnn : rnns_)
      temp.push_back(rnn->last().back());
    return temp;
  }

};

}
}