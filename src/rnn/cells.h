#pragma once

#include "marian.h"

#include "layers/generic.h"
#include "rnn/types.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <iomanip>
#include <string>

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
  Tanh(Ptr<ExpressionGraph> graph, Ptr<Options> options) : Cell(options) {
    int dimInput = options_->get<int>("dimInput");
    int dimState = options_->get<int>("dimState");
    std::string prefix = options_->get<std::string>("prefix");

    layerNorm_ = options_->get<bool>("layer-normalization", false);
    dropout_ = options_->get<float>("dropout", 0);

    U_ = graph->param(
        prefix + "_U", {dimState, dimState}, inits::glorot_uniform);

    if(dimInput)
      W_ = graph->param(
          prefix + "_W", {dimInput, dimState}, inits::glorot_uniform);

    b_ = graph->param(prefix + "_b", {1, dimState}, inits::zeros);

    if(dropout_ > 0.0f) {
      if(dimInput)
        dropMaskX_ = graph->dropout(dropout_, {1, dimInput});
      dropMaskS_ = graph->dropout(dropout_, {1, dimState});
    }

    if(layerNorm_) {
      if(dimInput)
        gamma1_ = graph->param(
            prefix + "_gamma1", {1, 3 * dimState}, inits::from_value(1.f));
      gamma2_ = graph->param(
          prefix + "_gamma2", {1, 3 * dimState}, inits::from_value(1.f));
    }
  }

  State apply(std::vector<Expr> inputs, State states, Expr mask = nullptr) {
    return applyState(applyInput(inputs), states, mask);
  }

  std::vector<Expr> applyInput(std::vector<Expr> inputs) override {
    Expr input;
    if(inputs.size() == 0)
      return {};
    else if(inputs.size() > 1)
      input = concatenate(inputs, keywords::axis = -1);
    else
      input = inputs.front();

    if(dropMaskX_)
      input = dropout(input, dropMaskX_);

    auto xW = dot(input, W_);

    if(layerNorm_)
      xW = layerNorm(xW, gamma1_);

    return {xW};
  }

  State applyState(std::vector<Expr> xWs, State state, Expr mask = nullptr) override {
    Expr recState = state.output;

    auto stateDropped = recState;
    if(dropMaskS_)
      stateDropped = dropout(recState, dropMaskS_);
    auto sU = dot(stateDropped, U_);
    if(layerNorm_)
      sU = layerNorm(sU, gamma2_);

    Expr output;
    if(xWs.empty())
      output = tanh(sU, b_);
    else {
      output = tanh(xWs.front(), sU, b_);
    }
    if(mask)
      return {output * mask, nullptr};
    else
      return {output, state.cell};
  }
};

/******************************************************************************/

class ReLU : public Cell {
private:
  Expr U_, W_, b_;
  Expr gamma1_;
  Expr gamma2_;

  bool layerNorm_;
  float dropout_;

  Expr dropMaskX_;
  Expr dropMaskS_;

public:
  ReLU(Ptr<ExpressionGraph> graph, Ptr<Options> options) : Cell(options) {
    int dimInput = options_->get<int>("dimInput");
    int dimState = options_->get<int>("dimState");
    std::string prefix = options_->get<std::string>("prefix");

    layerNorm_ = options_->get<bool>("layer-normalization", false);
    dropout_ = options_->get<float>("dropout", 0);

    U_ = graph->param(prefix + "_U", {dimState, dimState}, inits::diag(1.f));

    if(dimInput)
      W_ = graph->param(
          prefix + "_W", {dimInput, dimState}, inits::glorot_uniform);

    b_ = graph->param(prefix + "_b", {1, dimState}, inits::zeros);

    if(dropout_ > 0.0f) {
      if(dimInput)
        dropMaskX_ = graph->dropout(dropout_, {1, dimInput});
      dropMaskS_ = graph->dropout(dropout_, {1, dimState});
    }

    if(layerNorm_) {
      if(dimInput)
        gamma1_ = graph->param(prefix + "_gamma1", {1, dimState}, inits::ones);
      gamma2_ = graph->param(prefix + "_gamma2", {1, dimState}, inits::ones);
    }
  }

  State apply(std::vector<Expr> inputs, State states, Expr mask = nullptr) {
    return applyState(applyInput(inputs), states, mask);
  }

  std::vector<Expr> applyInput(std::vector<Expr> inputs) override {
    Expr input;
    if(inputs.size() == 0)
      return {};
    else if(inputs.size() > 1)
      input = concatenate(inputs, keywords::axis = -1);
    else
      input = inputs.front();

    if(dropMaskX_)
      input = dropout(input, dropMaskX_);

    auto xW = dot(input, W_);

    if(layerNorm_)
      xW = layerNorm(xW, gamma1_);

    return {xW};
  }

  State applyState(std::vector<Expr> xWs, State state, Expr mask = nullptr) override {
    Expr recState = state.output;

    auto stateDropped = recState;
    if(dropMaskS_)
      stateDropped = dropout(recState, dropMaskS_);
    auto sU = dot(stateDropped, U_);
    if(layerNorm_)
      sU = layerNorm(sU, gamma2_);

    Expr output;
    if(xWs.empty())
      output = relu(sU + b_);
    else {
      output = relu(xWs.front() + sU + b_);
    }
    if(mask)
      return {output * mask, state.cell};
    else
      return {output, state.cell};
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
  GRU(Ptr<ExpressionGraph> graph, Ptr<Options> options) : Cell(options) {
    int dimInput = opt<int>("dimInput");
    int dimState = opt<int>("dimState");
    std::string prefix = opt<std::string>("prefix");

    layerNorm_ = opt<bool>("layer-normalization", false);
    dropout_ = opt<float>("dropout", 0);
    final_ = opt<bool>("final", false);

    auto U = graph->param(
        prefix + "_U", {dimState, 2 * dimState}, inits::glorot_uniform);
    auto Ux = graph->param(
        prefix + "_Ux", {dimState, dimState}, inits::glorot_uniform);
    U_ = concatenate({U, Ux}, keywords::axis = -1);

    if(dimInput > 0) {
      auto W = graph->param(
          prefix + "_W", {dimInput, 2 * dimState}, inits::glorot_uniform);
      auto Wx = graph->param(
          prefix + "_Wx", {dimInput, dimState}, inits::glorot_uniform);
      W_ = concatenate({W, Wx}, keywords::axis = -1);
    }

    auto b = graph->param(prefix + "_b", {1, 2 * dimState}, inits::zeros);
    auto bx = graph->param(prefix + "_bx", {1, dimState}, inits::zeros);
    b_ = concatenate({b, bx}, keywords::axis = -1);

    // @TODO use this and adjust Amun model type saving and loading
    // U_ = graph->param(prefix + "_U", {dimState, 3 * dimState},
    //                  (Expr a) : UnaryNodeOp(a)inits::glorot_uniform);
    // W_ = graph->param(prefix + "_W", {dimInput, 3 * dimState},
    //                  (Expr a) : UnaryNodeOp(a)inits::glorot_uniform);
    // b_ = graph->param(prefix + "_b", {1, 3 * dimState},
    //                  (Expr a) : UnaryNodeOp(a)inits::zeros);

    if(dropout_ > 0.0f) {
      if(dimInput)
        dropMaskX_ = graph->dropout(dropout_, {1, dimInput});
      dropMaskS_ = graph->dropout(dropout_, {1, dimState});
    }

    if(layerNorm_) {
      if(dimInput)
        gamma1_ = graph->param(
            prefix + "_gamma1", {1, 3 * dimState}, inits::from_value(1.f));
      gamma2_ = graph->param(
          prefix + "_gamma2", {1, 3 * dimState}, inits::from_value(1.f));
    }
  }

  virtual State apply(std::vector<Expr> inputs,
                      State state,
                      Expr mask = nullptr) {
    return applyState(applyInput(inputs), state, mask);
  }

  virtual std::vector<Expr> applyInput(std::vector<Expr> inputs) override {
    Expr input;
    if(inputs.size() == 0)
      return {};
    else if(inputs.size() > 1)
      input = concatenate(inputs, keywords::axis = -1);
    else
      input = inputs[0];

    if(dropMaskX_)
      input = dropout(input, dropMaskX_);

    auto xW = dot(input, W_);
    if(layerNorm_)
      xW = layerNorm(xW, gamma1_);

    return {xW};
  }

  virtual State applyState(std::vector<Expr> xWs,
                           State state,
                           Expr mask = nullptr) override {
    auto stateOrig = state.output;
    auto stateDropped = stateOrig;
    if(dropMaskS_)
      stateDropped = dropout(stateOrig, dropMaskS_);

    auto sU = dot(stateDropped, U_);
    if(layerNorm_)
      sU = layerNorm(sU, gamma2_);

    Expr xW;
    if(xWs.empty()) {
      if(!fakeInput_ || fakeInput_->shape() != sU->shape())
        fakeInput_ = sU->graph()->constant(sU->shape(), inits::zeros);
      xW = fakeInput_;
    } else {
      xW = xWs.front();
    }

    auto output = mask ? gruOps({stateOrig, xW, sU, b_, mask}, final_)
                       : gruOps({stateOrig, xW, sU, b_}, final_);

    return {output, state.cell};  // no cell state, hence copy
  }
};

/**
 * @brief GRU unit supporting Nematus-like layer normalization.
 *
 * If no layer normalization is used, the unit is equivalent to
 * marian::rnn::GRU and uses the same kernel.
 *
 * @see marian::rnn::GRU
 */
class GRUNematus : public Cell {
protected:
  // Concatenated Us and Ws, used unless layer normalization enabled
  Expr UUx_, WWx_, bbx_;

  // Parameters used if layer normalization enabled
  Expr U_, W_, b_;
  Expr Ux_, Wx_, bx_;

  // Layer normalization parameters
  Expr W_lns_, W_lnb_;
  Expr Wx_lns_, Wx_lnb_;
  Expr U_lns_, U_lnb_;
  Expr Ux_lns_, Ux_lnb_;

  // Whether it is an encoder or decoder
  bool encoder_;
  // Whether it is an RNN final layer or hidden layer
  bool final_;
  // Whether it is a transition layer
  bool transition_;
  // Use layer normalization
  bool layerNorm_;

  // Dropout probability
  float dropout_;
  Expr dropMaskX_;
  Expr dropMaskS_;

  // Fake input with zeros replaces W and Wx in transition cells
  Expr fakeInput_;

public:
  GRUNematus(Ptr<ExpressionGraph> graph, Ptr<Options> options) : Cell(options) {
    int dimInput = opt<int>("dimInput");
    int dimState = opt<int>("dimState");

    auto prefix = opt<std::string>("prefix");
    encoder_ = prefix.substr(0, 7) == "encoder";
    transition_ = opt<bool>("transition", false);

    layerNorm_ = opt<bool>("layer-normalization", false);
    dropout_ = opt<float>("dropout", 0);
    final_ = opt<bool>("final", false);

    auto U = graph->param(
        prefix + "_U", {dimState, 2 * dimState}, inits::glorot_uniform);
    auto Ux = graph->param(
        prefix + "_Ux", {dimState, dimState}, inits::glorot_uniform);

    if(layerNorm_) {
      U_ = U;
      Ux_ = Ux;
    } else {
      UUx_ = concatenate({U, Ux}, keywords::axis = -1);
    }

    if(dimInput > 0) {
      auto W = graph->param(
          prefix + "_W", {dimInput, 2 * dimState}, inits::glorot_uniform);
      auto Wx = graph->param(
          prefix + "_Wx", {dimInput, dimState}, inits::glorot_uniform);
      if(layerNorm_) {
        W_ = W;
        Wx_ = Wx;
      } else {
        WWx_ = concatenate({W, Wx}, keywords::axis = -1);
      }
    }

    auto b = graph->param(prefix + "_b", {1, 2 * dimState}, inits::zeros);
    auto bx = graph->param(prefix + "_bx", {1, dimState}, inits::zeros);

    if(layerNorm_) {
      b_ = b;
      bx_ = bx;

      // in specific cases we need to pass bx to the kernel
      if(encoder_ && transition_) {
        auto b0 = graph->constant({1, 2 * dimState}, inits::zeros);
        bbx_ = concatenate({b0, bx}, keywords::axis = -1);
      } else {
        bbx_ = graph->constant({1, 3 * dimState}, inits::zeros);
      }
    } else {
      bbx_ = concatenate({b, bx}, keywords::axis = -1);
    }

    if(dropout_ > 0.0f) {
      if(dimInput)
        dropMaskX_ = graph->dropout(dropout_, {1, dimInput});
      dropMaskS_ = graph->dropout(dropout_, {1, dimState});
    }

    if(layerNorm_) {
      if(dimInput) {
        W_lns_ = graph->param(
            prefix + "_W_lns", {1, 2 * dimState}, inits::from_value(1.f));
        W_lnb_
            = graph->param(prefix + "_W_lnb", {1, 2 * dimState}, inits::zeros);
        Wx_lns_ = graph->param(
            prefix + "_Wx_lns", {1, 1 * dimState}, inits::from_value(1.f));
        Wx_lnb_
            = graph->param(prefix + "_Wx_lnb", {1, 1 * dimState}, inits::zeros);
      }
      U_lns_ = graph->param(
          prefix + "_U_lns", {1, 2 * dimState}, inits::from_value(1.f));
      U_lnb_ = graph->param(prefix + "_U_lnb", {1, 2 * dimState}, inits::zeros);
      Ux_lns_ = graph->param(
          prefix + "_Ux_lns", {1, 1 * dimState}, inits::from_value(1.f));
      Ux_lnb_
          = graph->param(prefix + "_Ux_lnb", {1, 1 * dimState}, inits::zeros);
    }
  }

  virtual State apply(std::vector<Expr> inputs,
                      State state,
                      Expr mask = nullptr) {
    return applyState(applyInput(inputs), state, mask);
  }

  virtual std::vector<Expr> applyInput(std::vector<Expr> inputs) override {
    Expr input;
    if(inputs.size() == 0)
      return {};
    else if(inputs.size() > 1)
      input = concatenate(inputs, keywords::axis = -1);
    else
      input = inputs[0];

    if(dropMaskX_)
      input = dropout(input, dropMaskX_);

    Expr xW;
    if(layerNorm_) {
      Expr W;   // RUH_1_ in Amun
      Expr Wx;  // RUH_2_ in Amun

      if(final_) {
        W = dot(input, W_);
        Wx = dot(input, Wx_);
      } else {
        W = affine(input, W_, b_);
        Wx = affine(input, Wx_, bx_);
      }
      W = layerNorm(W, W_lns_, W_lnb_, NEMATUS_LN_EPS);
      Wx = layerNorm(Wx, Wx_lns_, Wx_lnb_, NEMATUS_LN_EPS);

      xW = concatenate({W, Wx}, keywords::axis = -1);
    } else {
      xW = dot(input, WWx_);
    }

    return {xW};
  }

  virtual State applyState(std::vector<Expr> xWs,
                           State state,
                           Expr mask = nullptr) override {
    // make sure that we have transition layers
    assert(transition_ == xWs.empty());

    auto stateOrig = state.output;
    auto stateDropped = stateOrig;
    if(dropMaskS_)
      stateDropped = dropout(stateOrig, dropMaskS_);

    Expr sU;
    if(layerNorm_) {
      Expr U;   // Temp_1_ in Amun
      Expr Ux;  // Temp_2_ in Amun

      if(encoder_) {
        U = layerNorm(dot(stateDropped, U_), U_lns_, U_lnb_, NEMATUS_LN_EPS);
        Ux = layerNorm(
            dot(stateDropped, Ux_), Ux_lns_, Ux_lnb_, NEMATUS_LN_EPS);

        if(transition_) {
          U = U + b_;
        }
      } else {
        if(final_ || transition_) {
          U = affine(stateDropped, U_, b_);
          Ux = affine(stateDropped, Ux_, bx_);
        } else {
          U = dot(stateDropped, U_);
          Ux = dot(stateDropped, Ux_);
        }
        U = layerNorm(U, U_lns_, U_lnb_, NEMATUS_LN_EPS);
        Ux = layerNorm(Ux, Ux_lns_, Ux_lnb_, NEMATUS_LN_EPS);
      }

      sU = concatenate({U, Ux}, keywords::axis = -1);
    } else {
      sU = dot(stateDropped, UUx_);
    }

    Expr xW;
    if(transition_) {
      if(!fakeInput_ || fakeInput_->shape() != sU->shape())
        fakeInput_ = sU->graph()->constant(sU->shape(), inits::zeros);
      xW = fakeInput_;
    } else {
      xW = xWs.front();
    }

    Expr output;
    output = mask ? gruOps({stateOrig, xW, sU, bbx_, mask}, final_)
                  : gruOps({stateOrig, xW, sU, bbx_}, final_);

    return {output, state.cell};  // no cell state, hence copy
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
  FastLSTM(Ptr<ExpressionGraph> graph, Ptr<Options> options) : Cell(options) {
    int dimInput = opt<int>("dimInput");
    int dimState = opt<int>("dimState");
    std::string prefix = opt<std::string>("prefix");

    layerNorm_ = opt<bool>("layer-normalization", false);
    dropout_ = opt<float>("dropout", 0);

    U_ = graph->param(
        prefix + "_U", {dimState, 4 * dimState}, inits::glorot_uniform);
    if(dimInput)
      W_ = graph->param(
          prefix + "_W", {dimInput, 4 * dimState}, inits::glorot_uniform);

    b_ = graph->param(prefix + "_b", {1, 4 * dimState}, inits::zeros);

    if(dropout_ > 0.0f) {
      if(dimInput)
        dropMaskX_ = graph->dropout(dropout_, {1, dimInput});
      dropMaskS_ = graph->dropout(dropout_, {1, dimState});
    }

    if(layerNorm_) {
      if(dimInput)
        gamma1_ = graph->param(
            prefix + "_gamma1", {1, 4 * dimState}, inits::from_value(1.f));
      gamma2_ = graph->param(
          prefix + "_gamma2", {1, 4 * dimState}, inits::from_value(1.f));
    }
  }

  virtual State apply(std::vector<Expr> inputs,
                      State state,
                      Expr mask = nullptr) {
    return applyState(applyInput(inputs), state, mask);
  }

  virtual std::vector<Expr> applyInput(std::vector<Expr> inputs) override {
    Expr input;
    if(inputs.size() == 0)
      return {};
    else if(inputs.size() > 1) {
      input = concatenate(inputs, keywords::axis = -1);
    } else
      input = inputs.front();

    if(dropMaskX_)
      input = dropout(input, dropMaskX_);

    auto xW = dot(input, W_);

    if(layerNorm_)
      xW = layerNorm(xW, gamma1_);

    return {xW};
  }

  virtual State applyState(std::vector<Expr> xWs,
                           State state,
                           Expr mask = nullptr) override {
    auto recState = state.output;
    auto cellState = state.cell;

    auto recStateDropped = recState;
    if(dropMaskS_)
      recStateDropped = dropout(recState, dropMaskS_);

    auto sU = dot(recStateDropped, U_);

    if(layerNorm_)
      sU = layerNorm(sU, gamma2_);

    Expr xW;
    if(xWs.empty()) {
      if(!fakeInput_ || fakeInput_->shape() != sU->shape())
        fakeInput_ = sU->graph()->constant(sU->shape(), inits::zeros);
      xW = fakeInput_;
    } else {
      xW = xWs.front();
    }

    // dc/dp where p = W_i, U_i, ..., but without index o
    auto nextCellState = mask ? lstmOpsC({cellState, xW, sU, b_, mask})
                              : lstmOpsC({cellState, xW, sU, b_});

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
  Expr Um_, Wm_, bm_, bwm_;
  Expr gamma1m_, gamma2m_;

public:
  Multiplicative(Ptr<ExpressionGraph> graph, Ptr<Options> options)
      : CellType(graph, options) {
    int dimInput = options->get<int>("dimInput");
    int dimState = options->get<int>("dimState");
    std::string prefix = options->get<std::string>("prefix");

    Um_ = graph->param(
        prefix + "_Um", {dimState, dimState}, inits::glorot_uniform);
    Wm_ = graph->param(
        prefix + "_Wm", {dimInput, dimState}, inits::glorot_uniform);
    bm_ = graph->param(prefix + "_bm", {1, dimState}, inits::zeros);
    bwm_ = graph->param(prefix + "_bwm", {1, dimState}, inits::zeros);

    if(CellType::layerNorm_) {
      gamma1m_ = graph->param(
          prefix + "_gamma1m", {1, dimState}, inits::from_value(1.f));
      gamma2m_ = graph->param(
          prefix + "_gamma2m", {1, dimState}, inits::from_value(1.f));
    }
  }

  virtual std::vector<Expr> applyInput(std::vector<Expr> inputs) override {
    ABORT_IF(inputs.empty(), "Multiplicative LSTM expects input");

    Expr input;
    if(inputs.size() > 1) {
      input = concatenate(inputs, keywords::axis = -1);
    } else
      input = inputs.front();

    auto xWs = CellType::applyInput({input});
    auto xWm = affine(input, Wm_, bwm_);
    if(CellType::layerNorm_)
      xWm = layerNorm(xWm, gamma1m_);

    xWs.push_back(xWm);
    return xWs;
  }

  virtual State applyState(std::vector<Expr> xWs,
                           State state,
                           Expr mask = nullptr) override {
    auto xWm = xWs.back();
    xWs.pop_back();

    auto sUm = affine(state.output, Um_, bm_);
    if(CellType::layerNorm_)
      sUm = layerNorm(sUm, gamma2m_);

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
  SlowLSTM(Ptr<ExpressionGraph> graph, Ptr<Options> options) : Cell(options) {
    int dimInput = options_->get<int>("dimInput");
    int dimState = options_->get<int>("dimState");
    std::string prefix = options->get<std::string>("prefix");

    Uf_ = graph->param(
        prefix + "_Uf", {dimState, dimState}, inits::glorot_uniform);
    Wf_ = graph->param(
        prefix + "_Wf", {dimInput, dimState}, inits::glorot_uniform);
    bf_ = graph->param(prefix + "_bf", {1, dimState}, inits::zeros);

    Ui_ = graph->param(
        prefix + "_Ui", {dimState, dimState}, inits::glorot_uniform);
    Wi_ = graph->param(
        prefix + "_Wi", {dimInput, dimState}, inits::glorot_uniform);
    bi_ = graph->param(prefix + "_bi", {1, dimState}, inits::zeros);

    Uc_ = graph->param(
        prefix + "_Uc", {dimState, dimState}, inits::glorot_uniform);
    Wc_ = graph->param(
        prefix + "_Wc", {dimInput, dimState}, inits::glorot_uniform);
    bc_ = graph->param(prefix + "_bc", {1, dimState}, inits::zeros);

    Uo_ = graph->param(
        prefix + "_Uo", {dimState, dimState}, inits::glorot_uniform);
    Wo_ = graph->param(
        prefix + "_Wo", {dimInput, dimState}, inits::glorot_uniform);
    bo_ = graph->param(prefix + "_bo", {1, dimState}, inits::zeros);
  }

  State apply(std::vector<Expr> inputs, State state, Expr mask = nullptr) {
    return applyState(applyInput(inputs), state, mask);
  }

  std::vector<Expr> applyInput(std::vector<Expr> inputs) override {
    ABORT_IF(inputs.empty(), "Slow LSTM expects input");

    Expr input;
    if(inputs.size() > 1)
      input = concatenate(inputs, keywords::axis = -1);
    else
      input = inputs.front();

    auto xWf = dot(input, Wf_);
    auto xWi = dot(input, Wi_);
    auto xWo = dot(input, Wo_);
    auto xWc = dot(input, Wc_);

    return {xWf, xWi, xWo, xWc};
  }

  State applyState(std::vector<Expr> xWs, State state, Expr mask = nullptr) override {
    auto recState = state.output;
    auto cellState = state.cell;

    auto sUf = affine(recState, Uf_, bf_);
    auto sUi = affine(recState, Ui_, bi_);
    auto sUo = affine(recState, Uo_, bo_);
    auto sUc = affine(recState, Uc_, bc_);

    auto f = sigmoid(xWs[0] + sUf);
    auto i = sigmoid(xWs[1] + sUi);
    auto o = sigmoid(xWs[2] + sUo);
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
  TestLSTM(Ptr<ExpressionGraph> graph, Ptr<Options> options) : Cell(options) {
    int dimInput = options_->get<int>("dimInput");
    int dimState = options_->get<int>("dimState");
    std::string prefix = options->get<std::string>("prefix");

    auto Uf = graph->param(
        prefix + "_Uf", {dimState, dimState}, inits::glorot_uniform);
    auto Wf = graph->param(
        prefix + "_Wf", {dimInput, dimState}, inits::glorot_uniform);
    auto bf = graph->param(prefix + "_bf", {1, dimState}, inits::zeros);

    auto Ui = graph->param(
        prefix + "_Ui", {dimState, dimState}, inits::glorot_uniform);
    auto Wi = graph->param(
        prefix + "_Wi", {dimInput, dimState}, inits::glorot_uniform);
    auto bi = graph->param(prefix + "_bi", {1, dimState}, inits::zeros);

    auto Uc = graph->param(
        prefix + "_Uc", {dimState, dimState}, inits::glorot_uniform);
    auto Wc = graph->param(
        prefix + "_Wc", {dimInput, dimState}, inits::glorot_uniform);
    auto bc = graph->param(prefix + "_bc", {1, dimState}, inits::zeros);

    auto Uo = graph->param(
        prefix + "_Uo", {dimState, dimState}, inits::glorot_uniform);
    auto Wo = graph->param(
        prefix + "_Wo", {dimInput, dimState}, inits::glorot_uniform);
    auto bo = graph->param(prefix + "_bo", {1, dimState}, inits::zeros);

    U_ = concatenate({Uf, Ui, Uc, Uo}, keywords::axis = -1);
    W_ = concatenate({Wf, Wi, Wc, Wo}, keywords::axis = -1);
    b_ = concatenate({bf, bi, bc, bo}, keywords::axis = -1);
  }

  State apply(std::vector<Expr> inputs, State state, Expr mask = nullptr) {
    return applyState(applyInput(inputs), state, mask);
  }

  std::vector<Expr> applyInput(std::vector<Expr> inputs) override {
    ABORT_IF(inputs.empty(), "Test LSTM expects input");

    Expr input;
    if(inputs.size() > 1)
      input = concatenate(inputs, keywords::axis = -1);
    else
      input = inputs.front();

    auto xW = dot(input, W_);

    return {xW};
  }

  State applyState(std::vector<Expr> xWs, State state, Expr mask = nullptr) override {
    auto recState = state.output;
    auto cellState = state.cell;

    auto sU = dot(recState, U_);

    auto xW = xWs.front();

    // dc/dp where p = W_i, U_i, ..., but without index o
    auto nextCellState = mask ? lstmOpsC({cellState, xW, sU, b_, mask})
                              : lstmOpsC({cellState, xW, sU, b_});

    // dh/dp dh/dc where p = W_o, U_o, b_o
    auto nextRecState = mask ? lstmOpsO({nextCellState, xW, sU, b_, mask})
                             : lstmOpsO({nextCellState, xW, sU, b_});

    return {nextRecState, nextCellState};
  }
};

class SRU : public Cell {
private:
  Expr W_;
  Expr Wr_, br_;
  Expr Wf_, bf_;

  float dropout_;
  Expr dropMaskX_;

  float layerNorm_;
  Expr gamma_, gammaf_, gammar_;

public:
  SRU(Ptr<ExpressionGraph> graph, Ptr<Options> options) : Cell(options) {
    int dimInput = opt<int>("dimInput");
    int dimState = opt<int>("dimState");
    std::string prefix = opt<std::string>("prefix");

    ABORT_IF(dimInput != dimState,
             "For SRU state and input dims have to be equal");

    dropout_ = opt<float>("dropout", 0);
    layerNorm_ = opt<bool>("layer-normalization", false);

    W_ = graph->param(
        prefix + "_W", {dimInput, dimInput}, inits::glorot_uniform);

    Wf_ = graph->param(
        prefix + "_Wf", {dimInput, dimInput}, inits::glorot_uniform);
    bf_ = graph->param(prefix + "_bf", {1, dimInput}, inits::zeros);

    Wr_ = graph->param(
        prefix + "_Wr", {dimInput, dimInput}, inits::glorot_uniform);
    br_ = graph->param(prefix + "_br", {1, dimInput}, inits::zeros);

    if(dropout_ > 0.0f) {
      dropMaskX_ = graph->dropout(dropout_, {1, dimInput});
    }

    if(layerNorm_) {
      if(dimInput)
        gamma_ = graph->param(prefix + "_gamma", {1, dimState}, inits::ones);
      gammar_ = graph->param(prefix + "_gammar", {1, dimState}, inits::ones);
      gammaf_ = graph->param(prefix + "_gammaf", {1, dimState}, inits::ones);
    }
  }

  State apply(std::vector<Expr> inputs, State state, Expr mask = nullptr) {
    return applyState(applyInput(inputs), state, mask);
  }

  std::vector<Expr> applyInput(std::vector<Expr> inputs) override {
    ABORT_IF(inputs.empty(), "SRU expects input");

    Expr input;
    if(inputs.size() > 1)
      input = concatenate(inputs, keywords::axis = -1);
    else
      input = inputs.front();

    auto inputDropped = dropMaskX_ ? dropout(input, dropMaskX_) : input;

    Expr x, f, r;
    if(layerNorm_) {
      x = layerNorm(dot(inputDropped, W_), gamma_);
      f = layerNorm(dot(inputDropped, Wf_), gammaf_, bf_);
      r = layerNorm(dot(inputDropped, Wr_), gammar_, br_);
    } else {
      x = dot(inputDropped, W_);
      f = affine(inputDropped, Wf_, bf_);
      r = affine(inputDropped, Wr_, br_);
    }

    return {x, f, r, input};
  }

  State applyState(std::vector<Expr> xWs, State state, Expr mask = nullptr) override {
    auto recState = state.output;
    auto cellState = state.cell;

    auto x = xWs[0];
    auto f = xWs[1];
    auto r = xWs[2];
    auto input = xWs[3];

    auto nextCellState = highway(cellState, x, f);  // rename to "gate"?
    auto nextState = highway(tanh(nextCellState), input, r);

    auto maskedCellState = mask ? mask * nextCellState : nextCellState;
    auto maskedState = mask ? mask * nextState : nextState;

    return {maskedState, maskedCellState};
  }
};

class SSRU : public Cell {
private:
  Expr W_;
  Expr Wf_, bf_;

  float dropout_;
  Expr dropMaskX_;

  float layerNorm_;
  Expr gamma_, gammaf_;

public:
  SSRU(Ptr<ExpressionGraph> graph, Ptr<Options> options) : Cell(options) {
    int dimInput = options_->get<int>("dimInput");
    int dimState = options_->get<int>("dimState");

    std::string prefix = options->get<std::string>("prefix");

    ABORT_IF(dimInput != dimState,
             "For SSRU state and input dims have to be equal");

    dropout_ = opt<float>("dropout", 0);
    layerNorm_ = opt<bool>("layer-normalization", false);

    W_ = graph->param(
        prefix + "_W", {dimInput, dimInput}, inits::glorot_uniform);

    Wf_ = graph->param(
        prefix + "_Wf", {dimInput, dimInput}, inits::glorot_uniform);
    bf_ = graph->param(prefix + "_bf", {1, dimInput}, inits::zeros);

    if(dropout_ > 0.0f) {
      dropMaskX_ = graph->dropout(dropout_, {1, dimInput});
    }

    if(layerNorm_) {
      if(dimInput)
        gamma_ = graph->param(prefix + "_gamma", {1, dimState}, inits::ones);
      gammaf_ = graph->param(prefix + "_gammaf", {1, dimState}, inits::ones);
    }
  }

  State apply(std::vector<Expr> inputs, State state, Expr mask = nullptr) {
    return applyState(applyInput(inputs), state, mask);
  }

  std::vector<Expr> applyInput(std::vector<Expr> inputs) override {
    ABORT_IF(inputs.empty(), "SSRU expects input");

    Expr input;
    if(inputs.size() > 1)
      input = concatenate(inputs, keywords::axis = -1);
    else
      input = inputs.front();

    auto inputDropped = dropMaskX_ ? dropout(input, dropMaskX_) : input;

    Expr x, f;
    if(layerNorm_) {
      x = layerNorm(dot(inputDropped, W_), gamma_);
      f = layerNorm(dot(inputDropped, Wf_), gammaf_, bf_);
    } else {
      x = dot(inputDropped, W_);
      f = affine(inputDropped, Wf_, bf_);
    }

    return {x, f};
  }

  State applyState(std::vector<Expr> xWs, State state, Expr mask = nullptr) override {
    auto recState = state.output;
    auto cellState = state.cell;

    auto x = xWs[0];
    auto f = xWs[1];

    auto nextCellState = highway(cellState, x, f);  // rename to "gate"?
    auto nextState = relu(nextCellState);

    auto maskedCellState = mask ? mask * nextCellState : nextCellState;
    auto maskedState = mask ? mask * nextState : nextState;

    return {maskedState, maskedCellState};
  }
};

// class LSSRU : public Cell {
// private:
//   Expr W_;
//   Expr Wf_, bf_;

//   float dropout_;
//   Expr dropMaskX_;

// public:
//   LSSRU(Ptr<ExpressionGraph> graph, Ptr<Options> options) : Cell(options) {
//     int dimInput = options_->get<int>("dimInput");
//     int dimState = options_->get<int>("dimState");
//     std::string prefix = options->get<std::string>("prefix");

//     ABORT_IF(dimInput != dimState, "For SRU state and input dims have to be equal");

//     dropout_ = opt<float>("dropout", 0);

//     W_ = graph->param(prefix + "_W",
//                        {dimInput, dimInput},
//                        inits::glorot_uniform);

//     Wf_ = graph->param(prefix + "_Wf",
//                        {dimInput, dimInput},
//                        inits::glorot_uniform);
//     bf_ = graph->param(
//         prefix + "_bf", {1, dimInput}, inits::zeros);

//     if(dropout_ > 0.0f) {
//       dropMaskX_ = graph->dropout(dropout_, {1, dimInput});
//     }
//   }

//   State apply(std::vector<Expr> inputs, State state, Expr mask = nullptr) {
//     return applyState(applyInput(inputs), state, mask);
//   }

//   std::vector<Expr> applyInput(std::vector<Expr> inputs) {
//     ABORT_IF(inputs.empty(), "Slow SRU expects input");

//     Expr input;
//     if(inputs.size() > 1)
//       input = concatenate(inputs, keywords::axis = -1);
//     else
//       input = inputs.front();

//     auto inputDropped = dropMaskX_ ? dropout(input, dropMaskX_) : input;

//     auto x = dot(inputDropped, W_);
//     auto f = affine(inputDropped, Wf_, bf_);

//     return {x, f};
//   }

//   State applyState(std::vector<Expr> xWs, State state, Expr mask = nullptr) {
//     auto recState = state.output;
//     auto cellState = state.cell;

//     auto x = xWs[0];
//     auto f = xWs[1];

//     auto nextCellState = highwayLinear(cellState, x, f, 2.f); // rename to "gate"?
//     auto nextState = relu(nextCellState);
//     //auto nextState = nextCellState;

//     auto maskedCellState = mask ? mask * nextCellState : nextCellState;
//     auto maskedState = mask ? mask * nextState : nextState;

//     return {maskedState, maskedCellState};
//   }
// };

}  // namespace rnn
}  // namespace marian
