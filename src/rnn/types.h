#pragma once

#include <iostream>
#include <vector>

#include "common/definitions.h"
#include "graph/expression_graph.h"

namespace marian {
namespace rnn {

struct State {
  Expr output;
  Expr cell;

  State select(const std::vector<size_t>& indices) {

    int numSelected = indices.size();
    int dimState = output->shape()[1];

    if(cell) {
      return State{
        reshape(rows(output, indices), {1, dimState, 1, numSelected}),
        reshape(rows(cell, indices), {1, dimState, 1, numSelected})
      };
    }
    else {
      return State{
        reshape(rows(output, indices), {1, dimState, 1, numSelected}),
        nullptr
      };
    }
  }
};

class States {
  private:
    std::vector<State> states_;

  public:
    States() {}
    States(const std::vector<State>& states) : states_(states) {}

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

    State& operator[](size_t i) { return states_[i]; };
    const State& operator[](size_t i) const { return states_[i]; };

    State& back() { return states_.back(); }
    const State& back() const { return states_.back(); }

    State& front() { return states_.front(); }
    const State& front() const { return states_.front(); }

    size_t size() const { return states_.size(); };

    void push_back(const State& state) {
      states_.push_back(state);
    }

    States select(const std::vector<size_t>& indices) {
      States selected;
      for(auto& state : states_)
        selected.push_back(state.select(indices));
      return selected;
    }

    void reverse() {
      std::reverse(states_.begin(), states_.end());
    }

    void clear() { states_.clear(); }
};

class Cell;
struct CellInput;

struct Stackable : std::enable_shared_from_this<Stackable> {
  // required for dynamic_pointer_cast to detect polymorphism
  virtual ~Stackable() {}

  template <typename Cast>
  inline Ptr<Cast> as() {
    return std::dynamic_pointer_cast<Cast>(shared_from_this());
  }

  template <typename Cast>
  inline bool is() {
    return as<Cast>() != nullptr;
  }
};

struct CellInput : public Stackable {
  // Change this to apply(State)
  virtual Expr apply(State) = 0;
  virtual int dimOutput() = 0;
};

class Cell : public Stackable {
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

  State apply(std::vector<Expr> inputs, State state, Expr mask = nullptr) {
    return applyState(applyInput(inputs), state, mask);
  }

  virtual std::vector<Expr> applyInput(std::vector<Expr> inputs) = 0;
  virtual State applyState(std::vector<Expr>, State, Expr = nullptr) = 0;
};

class MultiCellInput : public CellInput {
private:
  std::vector<Ptr<CellInput>> inputs_;

public:
  MultiCellInput(const std::vector<Ptr<CellInput>>& inputs)
  : inputs_(inputs) {}

  void push_back(Ptr<CellInput> input) {
    inputs_.push_back(input);
  }

  virtual Expr apply(State state) {
    std::vector<Expr> outputs;
    for(auto input : inputs_)
      outputs.push_back(input->apply(state));

    if(outputs.size() > 1)
      return concatenate(outputs, keywords::axis = 1);
    else
      return outputs[0];
  }

  virtual int dimOutput() {
    int sum = 0;
    for(auto input : inputs_)
      sum += input->dimOutput();
    return sum;
  }
};

class StackedCell : public Cell {
private:
  std::vector<Ptr<Stackable>> stackables_;
  std::vector<Expr> lastInputs_;

public:
  StackedCell(int dimInput, int dimState) : Cell(dimInput, dimState) {}

  StackedCell(int dimInput, int dimState,
              const std::vector<Ptr<Stackable>>& stackables)
    : Cell(dimInput, dimState), stackables_(stackables) {}

  void push_back(Ptr<Stackable> stackable) {
    stackables_.push_back(stackable);
  }

  virtual std::vector<Expr> applyInput(std::vector<Expr> inputs) {
    lastInputs_ = inputs;
    return stackables_[0]->as<Cell>()->applyInput(inputs);
  }

  virtual State applyState(std::vector<Expr> mappedInputs, State state,
                           Expr mask = nullptr) {

    State hidden = stackables_[0]->as<Cell>()->applyState(mappedInputs, state, mask);;

    for(int i = 1; i < stackables_.size(); ++i) {
      if(stackables_[i]->is<Cell>()) {
        auto hiddenNext = stackables_[i]->as<Cell>()->apply(lastInputs_, hidden, mask);
        hidden = hiddenNext;
      }
      else {
        lastInputs_ = { stackables_[i]->as<CellInput>()->apply(hidden) };
      }
    }

    return hidden;
  };

  Ptr<Stackable> operator[](int i) {
    return stackables_[i];
  }

  Ptr<Stackable> at(int i) {
    return stackables_[i];
  }

};

}
}