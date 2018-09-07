#pragma once

#include "layers/generic.h"
#include "marian.h"
#include "rnn/cells.h"
#include "rnn/types.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <iomanip>
#include <string>

namespace marian {
namespace rnn {
enum struct dir : int {
  forward,
  backward,
  alternating_forward,
  alternating_backward
};
}
}  // namespace marian

YAML_REGISTER_TYPE(marian::rnn::dir, int)

namespace marian {
namespace rnn {

class BaseRNN {
protected:
  Ptr<ExpressionGraph> graph_;
  Ptr<Options> options_;

public:
  BaseRNN(Ptr<ExpressionGraph> graph, Ptr<Options> options)
      : graph_(graph), options_(options) {}

  virtual Expr transduce(Expr, Expr = nullptr) = 0;
  virtual Expr transduce(Expr, State, Expr = nullptr) = 0;
  virtual Expr transduce(Expr, States, Expr = nullptr) = 0;
  virtual States lastCellStates() = 0;
  virtual void push_back(Ptr<Cell>) = 0;
  virtual Ptr<Cell> at(int i) = 0;
  virtual Ptr<Options> getOptions() { return options_; }
};

class RNN;

class SingleLayerRNN : public BaseRNN {
private:
  Ptr<Cell> cell_;
  dir direction_;
  States last_;

  States apply(const Expr input,
               const States initialState,
               const Expr mask = nullptr) {
    last_.clear();

    State state = initialState.front();

    cell_->clear();

    auto xWs = cell_->applyInput({input});

    auto timeSteps = input->shape()[-3];

    States outputs;
    for(int i = 0; i < timeSteps; ++i) {
      int j = i;

      if(direction_ == dir::backward)
        j = timeSteps - i - 1;

      std::vector<Expr> steps(xWs.size());
      std::transform(xWs.begin(), xWs.end(), steps.begin(), [j](Expr e) {
        return step(e, j, -3);
      });

      if(mask)
        state = cell_->applyState(steps, state, step(mask, j, -3));
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

    int dimBatch = input->shape()[-2];
    int dimState = cell_->getOptions()->get<int>("dimState");

    auto output = graph->zeros({1, dimBatch, dimState});
    Expr cell = output;
    State startState{output, cell};

    return apply(input, States({startState}), mask);
  }

  SingleLayerRNN(Ptr<ExpressionGraph> graph, Ptr<Options> options)
      : BaseRNN(graph, options),
        direction_((dir)options->get<int>("direction", (int)dir::forward)) {}

public:
  friend RNN;

  // @TODO: benchmark whether this concatenation is a good idea
  virtual Expr transduce(Expr input, Expr mask = nullptr) override {
    return apply(input, mask).outputs();
  }

  virtual Expr transduce(Expr input, States states, Expr mask = nullptr) override {
    return apply(input, states, mask).outputs();
  }

  virtual Expr transduce(Expr input, State state, Expr mask = nullptr) override {
    return apply(input, States({state}), mask).outputs();
  }

  States lastCellStates() override { return last_; }

  void push_back(Ptr<Cell> cell) override { cell_ = cell; }

  virtual Ptr<Cell> at(int i) override {
    ABORT_IF(i > 0, "SingleRNN only has one cell");
    return cell_;
  }
};

class RNN : public BaseRNN, public std::enable_shared_from_this<RNN> {
private:
  bool skip_;
  bool skipFirst_;
  std::vector<Ptr<SingleLayerRNN>> rnns_;

public:
  RNN(Ptr<ExpressionGraph> graph, Ptr<Options> options)
      : BaseRNN(graph, options),
        skip_(options->get("skip", false)),
        skipFirst_(options->get("skipFirst", false)) {}

  void push_back(Ptr<Cell> cell) override {
    auto rnn
        = Ptr<SingleLayerRNN>(new SingleLayerRNN(graph_, cell->getOptions()));
    rnn->push_back(cell);
    rnns_.push_back(rnn);
  }

  Expr transduce(Expr input, Expr mask = nullptr) override {
    ABORT_IF(rnns_.empty(), "0 layers in RNN");

    Expr output;
    Expr layerInput = input;
    for(size_t i = 0; i < rnns_.size(); ++i) {
      auto lazyInput = layerInput;

      auto cell = rnns_[i]->at(0);
      auto lazyInputs = cell->getLazyInputs(shared_from_this());
      if(!lazyInputs.empty()) {
        lazyInputs.push_back(layerInput);
        lazyInput = concatenate(lazyInputs, keywords::axis = -1);
      }

      auto layerOutput = rnns_[i]->transduce(lazyInput, mask);

      if(skip_ && (skipFirst_ || i > 0))
        output = layerOutput + layerInput;
      else
        output = layerOutput;

      layerInput = output;
    }
    return output;
  }

  Expr transduce(Expr input, States states, Expr mask = nullptr) override {
    ABORT_IF(rnns_.empty(), "0 layers in RNN");

    Expr output;
    Expr layerInput = input;
    for(size_t i = 0; i < rnns_.size(); ++i) {
      Expr lazyInput;
      auto cell = rnns_[i]->at(0);
      auto lazyInputs = cell->getLazyInputs(shared_from_this());
      if(!lazyInputs.empty()) {
        lazyInputs.push_back(layerInput);
        lazyInput = concatenate(lazyInputs, keywords::axis = -1);
      } else {
        lazyInput = layerInput;
      }

      auto layerOutput
          = rnns_[i]->transduce(lazyInput, States({states[i]}), mask);

      if(skip_ && (skipFirst_ || i > 0))
        output = layerOutput + layerInput;
      else
        output = layerOutput;

      layerInput = output;
    }
    return output;
  }

  Expr transduce(Expr input, State state, Expr mask = nullptr) override {
    ABORT_IF(rnns_.empty(), "0 layers in RNN");

    Expr output;
    Expr layerInput = input;
    for(size_t i = 0; i < rnns_.size(); ++i) {
      auto lazyInput = layerInput;

      auto cell = rnns_[i]->at(0);
      auto lazyInputs = cell->getLazyInputs(shared_from_this());
      if(!lazyInputs.empty()) {
        lazyInputs.push_back(layerInput);
        lazyInput = concatenate(lazyInputs, keywords::axis = -1);
      }

      auto layerOutput = rnns_[i]->transduce(lazyInput, States({state}), mask);

      if(skip_ && (skipFirst_ || i > 0))
        output = layerOutput + layerInput;
      else
        output = layerOutput;

      layerInput = output;
    }
    return output;
  }

  States lastCellStates() override {
    States temp;
    for(auto rnn : rnns_)
      temp.push_back(rnn->lastCellStates().back());
    return temp;
  }

  virtual Ptr<Cell> at(int i) override { return rnns_[i]->at(0); }
};
}  // namespace rnn
}  // namespace marian
