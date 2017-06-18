#pragma once

#include "rnn/rnn.h"
#include "common/options.h"

namespace marian {
namespace rnn {

template <class Obj>
class Factory : public std::enable_shared_from_this<Factory<Obj>> {
protected:
  Ptr<Options> options_;
  Ptr<ExpressionGraph> graph_;

public:
  Factory(Ptr<ExpressionGraph> graph)
  : options_(New<Options>()), graph_(graph) {}

  virtual ~Factory() {}

  Ptr<Options> getOptions() {
    return options_;
  }

  std::string str() {
    return options_->str();
  }

  template <typename T>
  T get(const std::string& key) {
    return options_->get<T>(key);
  }
};

template <class Factory>
class Accumulator : public Factory {
public:
  Accumulator(Ptr<ExpressionGraph> graph) : Factory(graph) {}
  Accumulator(const Accumulator&) = default;
  Accumulator(Accumulator&&) = default;

  template <typename T>
  Accumulator& operator()(const std::string& key, T value) {
    Factory::getOptions()->set(key, value);
    return *this;
  }

  Accumulator& operator()(const std::string& yaml) {
    Factory::getOptions()->parse(yaml);
    return *this;
  }

  Accumulator& operator()(YAML::Node yaml) {
    Factory::getOptions()->merge(yaml);
    return *this;
  }
};

struct StackableFactory : public Factory<Stackable> {
  StackableFactory(Ptr<ExpressionGraph> graph) : Factory(graph) {}
  StackableFactory(const StackableFactory&) = default;
  StackableFactory(StackableFactory&&) = default;

  virtual ~StackableFactory() {}

  template <typename Cast>
  inline Ptr<Cast> as() {
    return std::dynamic_pointer_cast<Cast>(shared_from_this());
  }

  template <typename Cast>
  inline bool is() {
    return as<Cast>() != nullptr;
  }
};

struct InputFactory : public StackableFactory {
  InputFactory(Ptr<ExpressionGraph> graph) : StackableFactory(graph) {}
  virtual Ptr<CellInput> construct() = 0;
};

class CellFactory : public StackableFactory {
public:
  CellFactory(Ptr<ExpressionGraph> graph) : StackableFactory(graph) {}

  virtual Ptr<Cell> construct() {
    std::string type = options_->get<std::string>("type");
    if(type == "gru")
      return New<GRU>(graph_, options_);
    if(type == "lstm")
      return New<LSTM>(graph_, options_);
    if(type == "mlstm")
      return New<MLSTM>(graph_, options_);
    if(type == "mgru")
      return New<MGRU>(graph_, options_);
    if(type == "tanh")
      return New<Tanh>(graph_, options_);
    return New<GRU>(graph_, options_);
  }
};

typedef Accumulator<CellFactory> cell;

class StackedCellFactory : public CellFactory {
protected:
  std::vector<Ptr<StackableFactory>> stackableFactories_;

public:
  StackedCellFactory(Ptr<ExpressionGraph> graph) : CellFactory(graph) {}

  Ptr<Cell> construct() {
    auto stacked = New<StackedCell>(graph_, options_);
    for(auto sf : stackableFactories_) {
      if(sf->is<CellFactory>()) {
        auto cellFactory = sf->as<CellFactory>();
        cellFactory->getOptions()->merge(options_);
        stacked->push_back(cellFactory->construct());
      }
      else {
        auto inputFactory = sf->as<InputFactory>();
        inputFactory->getOptions()->merge(options_);
        stacked->push_back(inputFactory->construct());
      }
    }
    return stacked;
  }

  template <class F>
  void push_back(F& f) {
    stackableFactories_.push_back(New<F>(f));
  }
};

typedef Accumulator<StackedCellFactory> stacked_cell;

class AttentionFactory : public InputFactory {
protected:
  Ptr<EncoderState> state_;

public:
  AttentionFactory(Ptr<ExpressionGraph> graph) : InputFactory(graph) {}

  Ptr<CellInput> construct() {
    UTIL_THROW_IF2(!state_, "EncoderState not set");
    return New<Attention>(graph_, options_, state_);
  }

  void set_state(Ptr<EncoderState> state) {
    state_ = state;
  }

  int dimAttended() {
    UTIL_THROW_IF2(!state_, "EncoderState not set");
    return state_->getAttended()->shape()[1];
  }
};

typedef Accumulator<AttentionFactory> attention;

class cells {
private:
  std::string type_;
  size_t layers_;

public:
  cells(const std::string& type, size_t layers)
  : type_(type), layers_(layers) {}

  template <typename ...Args>
  std::vector<Ptr<Cell>> operator()(Ptr<ExpressionGraph> graph,
                                    std::string prefix,
                                    int dimInput,
                                    int dimState,
                                    Args ...args) {
    std::vector<Ptr<Cell>> cells;
    for(int i = 0; i < layers_; ++i)
      cells.push_back(cell(type_)(graph,
                                 prefix + "_l" + std::to_string(i),
                                 i == 0 ? dimInput : dimState,
                                 dimState,
                                 args...));
    return cells;
  }
};

}
}