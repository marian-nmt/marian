#pragma once

#include "rnn/rnn.h"
#include "layers/factory.h"

namespace marian {
namespace rnn {

struct StackableFactory : public Factory {
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
protected:
  std::vector<std::function<Expr(Ptr<rnn::RNN>)>> inputs_;

public:
  CellFactory(Ptr<ExpressionGraph> graph) : StackableFactory(graph) {}

  virtual Ptr<Cell> construct() {
    std::string type = options_->get<std::string>("type");
    if(type == "gru") {
      auto cell = New<GRU>(graph_, options_);
      cell->setLazyInputs(inputs_);
      return cell;
    } else if(type == "lstm") {
      auto cell = New<LSTM>(graph_, options_);
      cell->setLazyInputs(inputs_);
      return cell;
    } else if(type == "mlstm") {
      auto cell = New<MLSTM>(graph_, options_);
      cell->setLazyInputs(inputs_);
      return cell;
    } else if(type == "mgru"){
      auto cell = New<MGRU>(graph_, options_);
      cell->setLazyInputs(inputs_);
      return cell;
    } else if(type == "tanh") {
      auto cell = New<Tanh>(graph_, options_);
      cell->setLazyInputs(inputs_);
      return cell;
    } else {
      UTIL_THROW2("Unknown RNN cell type");
    }
  }

  CellFactory clone() {
    CellFactory aClone(graph_);
    aClone.options_->merge(options_);
    aClone.inputs_ = inputs_;
    return aClone;
  }

  virtual void add_input(std::function<Expr(Ptr<rnn::RNN>)> func) {
    inputs_.push_back(func);
  }

  virtual void add_input(Expr input) {
    inputs_.push_back([input](Ptr<rnn::RNN> rnn) { return input; });
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

    int lastDimInput = options_->get<int>("dimInput");

    for(int i = 0; i < stackableFactories_.size(); ++i) {
      auto sf = stackableFactories_[i];

      if(sf->is<CellFactory>()) {
        auto cellFactory = sf->as<CellFactory>();
        cellFactory->getOptions()->merge(options_);

        sf->getOptions()->set("dimInput", lastDimInput);
        lastDimInput = 0;

        if(i == 0)
          for(auto f : inputs_)
            cellFactory->add_input(f);

        stacked->push_back(cellFactory->construct());
      }
      else {
        auto inputFactory = sf->as<InputFactory>();
        inputFactory->getOptions()->merge(options_);
        auto input = inputFactory->construct();
        stacked->push_back(input);
        lastDimInput += input->dimOutput();
      }
    }
    return stacked;
  }

  template <class F>
  Accumulator<StackedCellFactory> push_back(const F& f) {
    stackableFactories_.push_back(New<F>(f));
    return Accumulator<StackedCellFactory>(*this);
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

  Accumulator<AttentionFactory> set_state(Ptr<EncoderState> state) {
    state_ = state;
    return Accumulator<AttentionFactory>(*this);
  }

  int dimAttended() {
    UTIL_THROW_IF2(!state_, "EncoderState not set");
    return state_->getAttended()->shape()[1];
  }
};

typedef Accumulator<AttentionFactory> attention;

class RNNFactory : public Factory {
protected:
  std::vector<Ptr<CellFactory>> layerFactories_;

public:
  RNNFactory(Ptr<ExpressionGraph> graph) : Factory(graph) {}

  Ptr<RNN> construct() {
    auto rnn = New<RNN>(graph_, options_);
    for(int i = 0; i < layerFactories_.size(); ++i) {
      auto lf = layerFactories_[i];

      lf->getOptions()->merge(options_);
      if(i > 0) {
        int dimInput = layerFactories_[i - 1]->getOptions()->get<int>("dimState")
          + lf->getOptions()->get<int>("dimInputExtra", 0);

        lf->getOptions()->set("dimInput", dimInput);
      }

      if(opt<rnn::dir>("direction", rnn::dir::forward) == rnn::dir::alternating_forward) {
        if(i % 2 == 0)
          lf->getOptions()->set("direction", rnn::dir::forward);
        else
          lf->getOptions()->set("direction", rnn::dir::backward);
      }

      if(opt<rnn::dir>("direction", rnn::dir::forward) == rnn::dir::alternating_backward) {
        if(i % 2 == 1)
          lf->getOptions()->set("direction", rnn::dir::forward);
        else
          lf->getOptions()->set("direction", rnn::dir::backward);
      }

      rnn->push_back(lf->construct());
    }
    return rnn;
  }

  Ptr<RNN> operator->() {
    return construct();
  }

  template <class F>
  Accumulator<RNNFactory> push_back(const F& f) {
    layerFactories_.push_back(New<F>(f));
    return Accumulator<RNNFactory>(*this);
  }

  RNNFactory clone() {
    RNNFactory aClone(graph_);
    aClone.options_->merge(options_);
    for(auto lf : layerFactories_)
      aClone.push_back(lf->clone());
    return aClone;
  }
};

typedef Accumulator<RNNFactory> rnn;

}
}
