#pragma once

#include "layers/factory.h"
#include "marian.h"
#include "rnn/rnn.h"

namespace marian {
/**
  *  The namespace rnn.
  *  Declare class Dense and all the available functions for creating
  *  <a href=https://en.wikipedia.org/wiki/Recurrent_neural_network>recurrent neural network (RNN)</a>
  *  network.
  */
namespace rnn {

typedef Factory StackableFactory;
//struct StackableFactory : public Factory {StackableFactory
//  using Factory::Factory;
//
//  virtual ~StackableFactory() {}
//
//  template <typename Cast>
//  inline Ptr<Cast> as() {
//    return std::dynamic_pointer_cast<Cast>(shared_from_this());
//  }
//
//  template <typename Cast>
//  inline bool is() {
//    return as<Cast>() != nullptr;
//  }
//};

struct InputFactory : public StackableFactory {
  virtual Ptr<CellInput> construct(Ptr<ExpressionGraph> graph) = 0;
};

/**
 * Base class for constructing RNN cells.
 * RNN cells only process a single timestep instead of the whole batches of input sequences.
 * There are nine types of RNN cells provided by Marian, i.e., `gru`, `gru-nematus`, `lstm`,
 * `mlstm`, `mgru`, `tanh`, `relu`, `sru`, `ssru`.
 */
class CellFactory : public StackableFactory {
protected:
  std::vector<std::function<Expr(Ptr<rnn::RNN>)>> inputs_;

public:
  virtual Ptr<Cell> construct(Ptr<ExpressionGraph> graph) {
    std::string type = options_->get<std::string>("type");
    if(type == "gru") {
      auto cell = New<GRU>(graph, options_);
      cell->setLazyInputs(inputs_);
      return cell;
    } else if(type == "gru-nematus") {
      auto cell = New<GRUNematus>(graph, options_);
      cell->setLazyInputs(inputs_);
      return cell;
    } else if(type == "lstm") {
      auto cell = New<LSTM>(graph, options_);
      cell->setLazyInputs(inputs_);
      return cell;
    } else if(type == "mlstm") {
      auto cell = New<MLSTM>(graph, options_);
      cell->setLazyInputs(inputs_);
      return cell;
    } else if(type == "mgru") {
      auto cell = New<MGRU>(graph, options_);
      cell->setLazyInputs(inputs_);
      return cell;
    } else if(type == "tanh") {
      auto cell = New<Tanh>(graph, options_);
      cell->setLazyInputs(inputs_);
      return cell;
    } else if(type == "relu") {
      auto cell = New<ReLU>(graph, options_);
      cell->setLazyInputs(inputs_);
      return cell;
    } else if(type == "sru") {
      auto cell = New<SRU>(graph, options_);
      cell->setLazyInputs(inputs_);
      return cell;
    } else if(type == "ssru") {
      auto cell = New<SSRU>(graph, options_);
      cell->setLazyInputs(inputs_);
      return cell;
    } else {
      ABORT("Unknown RNN cell type");
    }
  }

  CellFactory clone() {
    CellFactory aClone;
    aClone.options_->merge(options_);
    aClone.inputs_ = inputs_;
    return aClone;
  }

  virtual void add_input(std::function<Expr(Ptr<rnn::RNN>)> func) {
    inputs_.push_back(func);
  }

  virtual void add_input(Expr input) {
    inputs_.push_back([input](Ptr<rnn::RNN> /*rnn*/) { return input; });
  }
};

/** A convenience typedef for constructing RNN cells. */
typedef Accumulator<CellFactory> cell;

/** Base class for constructing a stack of RNN cells (`rnn::cell`). */
class StackedCellFactory : public CellFactory {
protected:
  std::vector<Ptr<StackableFactory>> stackableFactories_;

public:
  Ptr<Cell> construct(Ptr<ExpressionGraph> graph) override {
    auto stacked = New<StackedCell>(graph, options_);

    int lastDimInput = options_->get<int>("dimInput");

    for(size_t i = 0; i < stackableFactories_.size(); ++i) {
      auto sf = stackableFactories_[i];

      if(sf->is<CellFactory>()) {
        auto cellFactory = sf->as<CellFactory>();
        cellFactory->mergeOpts(options_);

        sf->setOpt("dimInput", lastDimInput);
        lastDimInput = 0;

        if(i == 0)
          for(auto f : inputs_)
            cellFactory->add_input(f);

        stacked->push_back(cellFactory->construct(graph));
      } else {
        auto inputFactory = sf->as<InputFactory>();
        inputFactory->mergeOpts(options_);
        auto input = inputFactory->construct(graph);
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

/** A convenience typedef for constructing a stack of RNN cells. */
typedef Accumulator<StackedCellFactory> stacked_cell;

/** Base class for constructing RNN layers. */
class RNNFactory : public Factory {
  using Factory::Factory;
protected:
  std::vector<Ptr<CellFactory>> layerFactories_;

public:
  Ptr<RNN> construct(Ptr<ExpressionGraph> graph) {
    auto rnn = New<RNN>(graph, options_);
    for(size_t i = 0; i < layerFactories_.size(); ++i) {
      auto lf = layerFactories_[i];

      lf->mergeOpts(options_);
      if(i > 0) {
        int dimInput
            = layerFactories_[i - 1]->opt<int>("dimState")
              + lf->opt<int>("dimInputExtra", 0);

        lf->setOpt("dimInput", dimInput);
      }

      if((rnn::dir)opt<int>("direction", (int)rnn::dir::forward)
         == rnn::dir::alternating_forward) {
        if(i % 2 == 0)
          lf->setOpt("direction", (int)rnn::dir::forward);
        else
          lf->setOpt("direction", (int)rnn::dir::backward);
      }

      if((rnn::dir)opt<int>("direction", (int)rnn::dir::forward)
         == rnn::dir::alternating_backward) {
        if(i % 2 == 1)
          lf->setOpt("direction", (int)rnn::dir::forward);
        else
          lf->setOpt("direction", (int)rnn::dir::backward);
      }

      rnn->push_back(lf->construct(graph));
    }
    return rnn;
  }

  template <class F>
  Accumulator<RNNFactory> push_back(const F& f) {
    layerFactories_.push_back(New<F>(f));
    return Accumulator<RNNFactory>(*this);
  }

  RNNFactory clone() {
    RNNFactory aClone;
    aClone.options_->merge(options_);
    for(auto lf : layerFactories_)
      aClone.push_back(lf->clone());
    return aClone;
  }
};

/** A convenience typedef for constructing RNN containers/layers. */
typedef Accumulator<RNNFactory> rnn;
}  // namespace rnn
}  // namespace marian
