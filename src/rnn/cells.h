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

#include "rnn/rnn.h"

namespace marian {
namespace rnn {

class Tanh : public Cell {
private:
  Expr U_, W_, b_;
  Expr gamma1_;
  Expr gamma2_;

  bool layerNorm_;
  float dropout_;

  Expr dropMaskX_;
  Expr dropMaskS_;

public:
  Tanh(Ptr<ExpressionGraph> graph, Ptr<Options> options)
    : Cell(options) {

    int dimInput = options_->get<int>("dimInput");
    int dimState = options_->get<int>("dimState");
    std::string prefix = options_->get<std::string>("prefix");

    layerNorm_ = options_->get<bool>("layer-normalization", false);
    dropout_ = options_->get<float>("dropout", 0);


    U_ = graph->param(prefix + "_U",
                      {dimState, dimState},
                      keywords::init = inits::glorot_uniform);

    if(dimInput)
      W_ = graph->param(prefix + "_W",
                        {dimInput, dimState},
                        keywords::init = inits::glorot_uniform);

    b_ = graph->param(
        prefix + "_b", {1, dimState}, keywords::init = inits::zeros);

    if(dropout_ > 0.0f) {
      if(dimInput)
        dropMaskX_ = graph->dropout(dropout_, {1, dimInput});
      dropMaskS_ = graph->dropout(dropout_, {1, dimState});
    }

    if(layerNorm_) {
      if(dimInput)
        gamma1_ = graph->param(prefix + "_gamma1",
                               {1, 3 * dimState},
                               keywords::init = inits::from_value(1.f));
      gamma2_ = graph->param(prefix + "_gamma2",
                             {1, 3 * dimState},
                             keywords::init = inits::from_value(1.f));
    }
  }

  State apply(std::vector<Expr> inputs,
              State states,
              Expr mask = nullptr) {
    return applyState(applyInput(inputs), states, mask);
  }

  std::vector<Expr> applyInput(std::vector<Expr> inputs) {
    Expr input;
    if(inputs.size() == 0)
      return {};
    else if(inputs.size() > 1)
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

  State applyState(std::vector<Expr> xWs,
                   State state,
                   Expr mask = nullptr) {
    Expr recState = state.output;

    auto stateDropped = recState;
    if(dropMaskS_)
      stateDropped = dropout(recState, keywords::mask = dropMaskS_);
    auto sU = dot(stateDropped, U_);
    if(layerNorm_)
      sU = layer_norm(sU, gamma2_);


    Expr output;
    if(xWs.empty())
      output = tanh(sU, b_);
    else {
      output = tanh(xWs.front(), sU, b_);
    }
    if(mask)
      return {output * mask, nullptr};
    else
      return { output, state.cell };
  }
};


/******************************************************************************/

Expr gruOps(const std::vector<Expr>& nodes, bool final = false);

class GRU : public Cell {
protected:
  std::string prefix_;

  Expr U_, W_, b_;
  Expr gamma1_;
  Expr gamma2_;

  bool final_;
  bool layerNorm_;
  float dropout_;

  Expr dropMaskX_;
  Expr dropMaskS_;

  Expr fakeInput_;

public:
  GRU(Ptr<ExpressionGraph> graph,
      Ptr<Options> options)
      : Cell(options) {

    int dimInput = opt<int>("dimInput");
    int dimState = opt<int>("dimState");
    std::string prefix = opt<std::string>("prefix");

    layerNorm_ = opt<bool>("layer-normalization", false);
    dropout_ = opt<float>("dropout", 0);
    final_ = opt<bool>("final", false);

    auto U = graph->param(prefix + "_U",
                          {dimState, 2 * dimState},
                          keywords::init = inits::glorot_uniform);
    auto Ux = graph->param(prefix + "_Ux",
                           {dimState, dimState},
                           keywords::init = inits::glorot_uniform);
    U_ = concatenate({U, Ux}, keywords::axis = 1);


    if(dimInput > 0) {
      auto W = graph->param(prefix + "_W",
                            {dimInput, 2 * dimState},
                            keywords::init = inits::glorot_uniform);
      auto Wx = graph->param(prefix + "_Wx",
                             {dimInput, dimState},
                             keywords::init = inits::glorot_uniform);
      W_ = concatenate({W, Wx}, keywords::axis = 1);
    }

    auto b = graph->param(
        prefix + "_b", {1, 2 * dimState}, keywords::init = inits::zeros);
    auto bx = graph->param(
        prefix + "_bx", {1, dimState}, keywords::init = inits::zeros);
    b_ = concatenate({b, bx}, keywords::axis = 1);

    // @TODO use this and adjust Amun model type saving and loading
    // U_ = graph->param(prefix + "_U", {dimState, 3 * dimState},
    //                  keywords::init=inits::glorot_uniform);
    // W_ = graph->param(prefix + "_W", {dimInput, 3 * dimState},
    //                  keywords::init=inits::glorot_uniform);
    // b_ = graph->param(prefix + "_b", {1, 3 * dimState},
    //                  keywords::init=inits::zeros);

    if(dropout_ > 0.0f) {
      if(dimInput)
        dropMaskX_ = graph->dropout(dropout_, {1, dimInput});
      dropMaskS_ = graph->dropout(dropout_, {1, dimState});
    }

    if(layerNorm_) {
      if(dimInput)
        gamma1_ = graph->param(prefix + "_gamma1",
                               {1, 3 * dimState},
                               keywords::init = inits::from_value(1.f));
      gamma2_ = graph->param(prefix + "_gamma2",
                             {1, 3 * dimState},
                             keywords::init = inits::from_value(1.f));
    }
  }

  virtual State apply(std::vector<Expr> inputs,
                      State state,
                      Expr mask = nullptr) {
    return applyState(applyInput(inputs), state, mask);
  }

  virtual std::vector<Expr> applyInput(std::vector<Expr> inputs) {
    Expr input;
    if(inputs.size() == 0)
      return {};
    else if(inputs.size() > 1)
      input = concatenate(inputs, keywords::axis = 1);
    else
      input = inputs[0];

    if(dropMaskX_)
      input = dropout(input, keywords::mask = dropMaskX_);

    auto xW = dot(input, W_);

    if(layerNorm_)
      xW = layer_norm(xW, gamma1_);
    return {xW};
  }

  virtual State applyState(std::vector<Expr> xWs,
                           State state,
                           Expr mask = nullptr) {

    auto stateOrig = state.output;
    auto stateDropped = stateOrig;
    if(dropMaskS_)
      stateDropped = dropout(stateOrig, keywords::mask = dropMaskS_);

    auto sU = dot(stateDropped, U_);

    if(layerNorm_)
      sU = layer_norm(sU, gamma2_);

    Expr xW;
    if(xWs.empty()) {
      if(not fakeInput_)
        fakeInput_ = sU->graph()->constant(sU->shape(), keywords::init=inits::zeros);
      xW = fakeInput_;
    }
    else {
      xW = xWs.front();
    }

    auto output = mask ? gruOps({stateOrig, xW, sU, b_, mask}, final_) :
                         gruOps({stateOrig, xW, sU, b_}, final_);

    return { output, state.cell }; // no cell state, hence copy
  }
};

/******************************************************************************/

Expr lstmOpsC(const std::vector<Expr>& nodes);
Expr lstmOpsO(const std::vector<Expr>& nodes);

class FastLSTM : public Cell {
protected:
  std::string prefix_;

  Expr U_, W_, b_;
  Expr gamma1_;
  Expr gamma2_;

  bool layerNorm_;
  float dropout_;

  Expr dropMaskX_;
  Expr dropMaskS_;

  Expr fakeInput_;

public:
  FastLSTM(Ptr<ExpressionGraph> graph,
           Ptr<Options> options)
      : Cell(options) {

    int dimInput = opt<int>("dimInput");
    int dimState = opt<int>("dimState");
    std::string prefix = opt<std::string>("prefix");

    layerNorm_ = opt<bool>("layer-normalization", false);
    dropout_ = opt<float>("dropout", 0);

    U_ = graph->param(prefix + "_U", {dimState, 4 * dimState},
                      keywords::init=inits::glorot_uniform);
    if(dimInput)
      W_ = graph->param(prefix + "_W", {dimInput, 4 * dimState},
                        keywords::init=inits::glorot_uniform);

    b_ = graph->param(prefix + "_b", {1, 4 * dimState},
                      keywords::init=inits::zeros);

    if(dropout_ > 0.0f) {
      if(dimInput)
        dropMaskX_ = graph->dropout(dropout_, {1, dimInput});
      dropMaskS_ = graph->dropout(dropout_, {1, dimState});
    }

    if(layerNorm_) {
      if(dimInput)
        gamma1_ = graph->param(prefix + "_gamma1",
                               {1, 4 * dimState},
                               keywords::init = inits::from_value(1.f));
      gamma2_ = graph->param(prefix + "_gamma2",
                             {1, 4 * dimState},
                             keywords::init = inits::from_value(1.f));
    }
  }

  virtual State apply(std::vector<Expr> inputs,
                      State state,
                      Expr mask = nullptr) {
    return applyState(applyInput(inputs), state, mask);
  }

  virtual std::vector<Expr> applyInput(std::vector<Expr> inputs) {
    Expr input;
    if(inputs.size() == 0)
      return {};
    else if(inputs.size() > 1)
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

  virtual State applyState(std::vector<Expr> xWs,
                           State state,
                           Expr mask = nullptr) {

    auto recState = state.output;
    auto cellState = state.cell;

    auto recStateDropped = recState;
    if(dropMaskS_)
      recStateDropped = dropout(recState, keywords::mask = dropMaskS_);

    auto sU = dot(recStateDropped, U_);

    if(layerNorm_)
      sU = layer_norm(sU, gamma2_);

    Expr xW;
    if(xWs.empty()) {
      if(not fakeInput_)
        fakeInput_ = sU->graph()->constant(sU->shape(),
                                           keywords::init=inits::zeros);
      xW = fakeInput_;
    }
    else {
      xW = xWs.front();
    }

    // dc/dp where p = W_i, U_i, ..., but without index o
    auto nextCellState = mask ?
      lstmOpsC({cellState, xW, sU, b_, mask}) :
      lstmOpsC({cellState, xW, sU, b_});

    // dh/dp dh/dc where p = W_o, U_o, b_o
    auto nextRecState = lstmOpsO({nextCellState, xW, sU, b_});

    return {nextRecState, nextCellState};
  }
};

using LSTM = FastLSTM;

/******************************************************************************/
// Experimentak cells, use with care

template <class CellType>
class Multiplicative : public CellType {
  protected:
    Expr Um_, Wm_, bm_;
    Expr gamma1m_, gamma2m_;

  public:
    Multiplicative(Ptr<ExpressionGraph> graph,
                   Ptr<Options> options)
      : CellType(graph, options) {

      int dimInput = options->get<int>("dimInput");
      int dimState = options->get<int>("dimState");
      std::string prefix = options->get<std::string>("prefix");

      Um_ = graph->param(prefix + "_Um", {dimState, dimState},
                         keywords::init=inits::glorot_uniform);
      Wm_ = graph->param(prefix + "_Wm", {dimInput, dimState},
                         keywords::init=inits::glorot_uniform);
      bm_ = graph->param(prefix + "_bm", {1, dimState},
                         keywords::init=inits::zeros);

      if(CellType::layerNorm_) {
        gamma1m_ = graph->param(prefix + "_gamma1m",
                                {1, dimState},
                                keywords::init = inits::from_value(1.f));
        gamma2m_ = graph->param(prefix + "_gamma2m",
                                {1, dimState},
                                keywords::init = inits::from_value(1.f));
      }
    }

  virtual std::vector<Expr> applyInput(std::vector<Expr> inputs) {
    UTIL_THROW_IF2(inputs.empty(), "Multiplicative LSTM expects input");

    Expr input;
    if(inputs.size() > 1)
      input = concatenate(inputs, keywords::axis = 1);
    else
      input = inputs.front();

    auto xWs = CellType::applyInput({input});
    auto xWm = dot(input, Wm_);
    if(CellType::layerNorm_)
      xWm = layer_norm(xWm, gamma1m_);

    xWs.push_back(xWm);
    return xWs;
  }

  virtual State applyState(std::vector<Expr> xWs,
                              State state,
                              Expr mask = nullptr) {
    auto xWm = xWs.back();
    xWs.pop_back();

    auto sUm = affine(state.output, Um_, bm_);
    if(CellType::layerNorm_)
      sUm = layer_norm(sUm, gamma2m_);

    auto mstate = xWm * sUm;

    return CellType::applyState(xWs, State({mstate, state.cell}), mask);
  }
};

using MLSTM = Multiplicative<LSTM>;
using MGRU = Multiplicative<GRU>;

/******************************************************************************/
// SlowLSTM and TestLSTM are for comparing efficient kernels for gradients with
// naive but correct LSTM version.

class SlowLSTM : public Cell {
private:

  Expr Uf_, Wf_, bf_;
  Expr Ui_, Wi_, bi_;
  Expr Uo_, Wo_, bo_;
  Expr Uc_, Wc_, bc_;

public:
  SlowLSTM(Ptr<ExpressionGraph> graph,
           Ptr<Options> options)
      : Cell(options) {

    int dimInput = options_->get<int>("dimInput");
    int dimState = options_->get<int>("dimState");
    std::string prefix = options->get<std::string>("prefix");

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

    Uc_ = graph->param(prefix + "_Uc", {dimState, dimState},
                       keywords::init=inits::glorot_uniform);
    Wc_ = graph->param(prefix + "_Wc", {dimInput, dimState},
                       keywords::init=inits::glorot_uniform);
    bc_ = graph->param(prefix + "_bc", {1, dimState},
                       keywords::init=inits::zeros);

    Uo_ = graph->param(prefix + "_Uo", {dimState, dimState},
                       keywords::init=inits::glorot_uniform);
    Wo_ = graph->param(prefix + "_Wo", {dimInput, dimState},
                       keywords::init=inits::glorot_uniform);
    bo_ = graph->param(prefix + "_bo", {1, dimState},
                       keywords::init=inits::zeros);

  }

  State apply(std::vector<Expr> inputs,
                 State state,
                 Expr mask = nullptr) {
    return applyState(applyInput(inputs), state, mask);
  }

  std::vector<Expr> applyInput(std::vector<Expr> inputs) {
    UTIL_THROW_IF2(inputs.empty(), "Slow LSTM expects input");

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

  State applyState(std::vector<Expr> xWs,
                      State state,
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
};

/******************************************************************************/

class TestLSTM : public Cell {
private:
  Expr U_, W_, b_;

public:
  TestLSTM(Ptr<ExpressionGraph> graph, Ptr<Options> options)
      : Cell(options) {

    int dimInput = options_->get<int>("dimInput");
    int dimState = options_->get<int>("dimState");
    std::string prefix = options->get<std::string>("prefix");

    auto Uf = graph->param(prefix + "_Uf", {dimState, dimState},
                       keywords::init=inits::glorot_uniform);
    auto Wf = graph->param(prefix + "_Wf", {dimInput, dimState},
                       keywords::init=inits::glorot_uniform);
    auto bf = graph->param(prefix + "_bf", {1, dimState},
                       keywords::init=inits::zeros);

    auto Ui = graph->param(prefix + "_Ui", {dimState, dimState},
                       keywords::init=inits::glorot_uniform);
    auto Wi = graph->param(prefix + "_Wi", {dimInput, dimState},
                       keywords::init=inits::glorot_uniform);
    auto bi = graph->param(prefix + "_bi", {1, dimState},
                       keywords::init=inits::zeros);

    auto Uc = graph->param(prefix + "_Uc", {dimState, dimState},
                       keywords::init=inits::glorot_uniform);
    auto Wc = graph->param(prefix + "_Wc", {dimInput, dimState},
                       keywords::init=inits::glorot_uniform);
    auto bc = graph->param(prefix + "_bc", {1, dimState},
                       keywords::init=inits::zeros);

    auto Uo = graph->param(prefix + "_Uo", {dimState, dimState},
                       keywords::init=inits::glorot_uniform);
    auto Wo = graph->param(prefix + "_Wo", {dimInput, dimState},
                       keywords::init=inits::glorot_uniform);
    auto bo = graph->param(prefix + "_bo", {1, dimState},
                       keywords::init=inits::zeros);

    U_ = concatenate({Uf, Ui, Uc, Uo}, keywords::axis = 1);
    W_ = concatenate({Wf, Wi, Wc, Wo}, keywords::axis = 1);
    b_ = concatenate({bf, bi, bc, bo}, keywords::axis = 1);

  }

  State apply(std::vector<Expr> inputs,
                 State state,
                 Expr mask = nullptr) {
    return applyState(applyInput(inputs), state, mask);
  }

  std::vector<Expr> applyInput(std::vector<Expr> inputs) {
    UTIL_THROW_IF2(inputs.empty(), "Test LSTM expects input");

    Expr input;
    if(inputs.size() > 1)
      input = concatenate(inputs, keywords::axis = 1);
    else
      input = inputs.front();

    auto xW = dot(input, W_);

    return {xW};
  }

  State applyState(std::vector<Expr> xWs,
                      State state,
                      Expr mask = nullptr) {

    auto recState = state.output;
    auto cellState = state.cell;

    auto sU = dot(recState, U_);

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
};

}
}
