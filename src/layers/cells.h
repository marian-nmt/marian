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
#include "layers/rnn.h"

namespace marian {


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
  template <typename... Args>
  Tanh(Ptr<ExpressionGraph> graph,
       const std::string prefix,
       int dimInput,
       int dimState,
       Args... args) : Cell(dimInput, dimState) {
    U_ = graph->param(prefix + "_U",
                      {dimState, dimState},
                      keywords::init = inits::glorot_uniform);
    W_ = graph->param(prefix + "_W",
                      {dimInput, dimState},
                      keywords::init = inits::glorot_uniform);
    b_ = graph->param(
        prefix + "_b", {1, dimState}, keywords::init = inits::zeros);

    layerNorm_ = Get(keywords::normalize, false, args...);

    dropout_ = Get(keywords::dropout_prob, 0.0f, args...);
    if(dropout_ > 0.0f) {
      dropMaskX_ = graph->dropout(dropout_, {1, dimInput});
      dropMaskS_ = graph->dropout(dropout_, {1, dimState});
    }

    if(layerNorm_) {
      gamma1_ = graph->param(prefix + "_gamma1",
                             {1, 3 * dimState},
                             keywords::init = inits::from_value(1.f));
      gamma2_ = graph->param(prefix + "_gamma2",
                             {1, 3 * dimState},
                             keywords::init = inits::from_value(1.f));
    }
  }

  RNNState apply(std::vector<Expr> inputs,
                 RNNState states,
                 Expr mask = nullptr) {
    return applyState(applyInput(inputs), states, mask);
  }

  std::vector<Expr> applyInput(std::vector<Expr> inputs) {
    Expr input;
    if(inputs.size() > 1)
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

  RNNState applyState(std::vector<Expr> xWs,
                      RNNState state,
                      Expr mask = nullptr) {
    Expr recState = state.output;

    auto stateDropped = recState;
    if(dropMaskS_)
      stateDropped = dropout(recState, keywords::mask = dropMaskS_);
    auto sU = dot(stateDropped, U_);
    if(layerNorm_)
      sU = layer_norm(sU, gamma2_);

    auto xW = xWs.front();

    auto output = tanh(xW, sU, b_);
    if(mask)
      return {output * mask, nullptr};
    else
      return {output, nullptr};
  }

  size_t numStates() { return 1; }
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

public:
  template <typename... Args>
  GRU(Ptr<ExpressionGraph> graph,
      const std::string prefix,
      int dimInput,
      int dimState,
      Args... args)
      : Cell(dimInput, dimState), prefix_{prefix} {
    auto U = graph->param(prefix + "_U",
                          {dimState, 2 * dimState},
                          keywords::init = inits::glorot_uniform);
    auto W = graph->param(prefix + "_W",
                          {dimInput, 2 * dimState},
                          keywords::init = inits::glorot_uniform);
    auto b = graph->param(
        prefix + "_b", {1, 2 * dimState}, keywords::init = inits::zeros);
    auto Ux = graph->param(prefix + "_Ux",
                           {dimState, dimState},
                           keywords::init = inits::glorot_uniform);
    auto Wx = graph->param(prefix + "_Wx",
                           {dimInput, dimState},
                           keywords::init = inits::glorot_uniform);
    auto bx = graph->param(
        prefix + "_bx", {1, dimState}, keywords::init = inits::zeros);

    U_ = concatenate({U, Ux}, keywords::axis = 1);
    W_ = concatenate({W, Wx}, keywords::axis = 1);
    b_ = concatenate({b, bx}, keywords::axis = 1);

    // @TODO use this and adjust Amun model type saving and loading
    // U_ = graph->param(prefix + "_U", {dimState, 3 * dimState},
    //                  keywords::init=inits::glorot_uniform);
    // W_ = graph->param(prefix + "_W", {dimInput, 3 * dimState},
    //                  keywords::init=inits::glorot_uniform);
    // b_ = graph->param(prefix + "_b", {1, 3 * dimState},
    //                  keywords::init=inits::zeros);

    final_ = Get(keywords::final, false, args...);
    layerNorm_ = Get(keywords::normalize, false, args...);

    dropout_ = Get(keywords::dropout_prob, 0.0f, args...);
    if(dropout_ > 0.0f) {
      dropMaskX_ = graph->dropout(dropout_, {1, dimInput});
      dropMaskS_ = graph->dropout(dropout_, {1, dimState});
    }

    if(layerNorm_) {
      gamma1_ = graph->param(prefix + "_gamma1",
                             {1, 3 * dimState},
                             keywords::init = inits::from_value(1.f));
      gamma2_ = graph->param(prefix + "_gamma2",
                             {1, 3 * dimState},
                             keywords::init = inits::from_value(1.f));
    }
  }

  virtual RNNState apply(std::vector<Expr> inputs,
                 RNNState state,
                 Expr mask = nullptr) {
    return applyState(applyInput(inputs), state, mask);
  }

  virtual std::vector<Expr> applyInput(std::vector<Expr> inputs) {
    Expr input;
    if(inputs.size() > 1)
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

  virtual RNNState applyState(std::vector<Expr> xWs,
                      RNNState state,
                      Expr mask = nullptr) {

    auto stateOrig = state.output;
    auto stateDropped = stateOrig;
    if(dropMaskS_)
      stateDropped = dropout(stateOrig, keywords::mask = dropMaskS_);

    auto sU = dot(stateDropped, U_);

    if(layerNorm_)
      sU = layer_norm(sU, gamma2_);

    auto xW = xWs.front();

    auto output = mask ? gruOps({stateOrig, xW, sU, b_, mask}, final_) :
                         gruOps({stateOrig, xW, sU, b_}, final_);

    return {output, nullptr}; // no cell state, hence nullptr
  }

  virtual size_t numStates() { return 1; }
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

public:
  template <typename... Args>
  FastLSTM(Ptr<ExpressionGraph> graph,
      const std::string prefix,
      int dimInput,
      int dimState,
      Args... args)
      : Cell(dimInput, dimState), prefix_{prefix} {

    U_ = graph->param(prefix + "_U", {dimState, 4 * dimState},
                      keywords::init=inits::glorot_uniform);
    W_ = graph->param(prefix + "_W", {dimInput, 4 * dimState},
                      keywords::init=inits::glorot_uniform);
    b_ = graph->param(prefix + "_b", {1, 4 * dimState},
                      keywords::init=inits::zeros);

    layerNorm_ = Get(keywords::normalize, false, args...);

    dropout_ = Get(keywords::dropout_prob, 0.0f, args...);
    if(dropout_ > 0.0f) {
      dropMaskX_ = graph->dropout(dropout_, {1, dimInput});
      dropMaskS_ = graph->dropout(dropout_, {1, dimState});
    }

    if(layerNorm_) {
      gamma1_ = graph->param(prefix + "_gamma1",
                             {1, 4 * dimState},
                             keywords::init = inits::from_value(1.f));
      gamma2_ = graph->param(prefix + "_gamma2",
                             {1, 4 * dimState},
                             keywords::init = inits::from_value(1.f));
    }
  }

  virtual RNNState apply(std::vector<Expr> inputs,
                 RNNState state,
                 Expr mask = nullptr) {
    return applyState(applyInput(inputs), state, mask);
  }

  virtual std::vector<Expr> applyInput(std::vector<Expr> inputs) {
    Expr input;
    if(inputs.size() > 1)
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

  virtual RNNState applyState(std::vector<Expr> xWs,
                              RNNState state,
                              Expr mask = nullptr) {

    auto recState = state.output;
    auto cellState = state.cell;

    auto recStateDropped = recState;
    if(dropMaskS_)
      recStateDropped = dropout(recState, keywords::mask = dropMaskS_);

    auto sU = dot(recStateDropped, U_);

    if(layerNorm_)
      sU = layer_norm(sU, gamma2_);

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

  virtual size_t numStates() { return 2; }
};

template <class CellType>
class Multiplicative : public CellType {
  private:
    Expr Um_, Wm_, bm_;
    Expr gamma1m_, gamma2m_;

  public:

    template <typename... Args>
    Multiplicative(Ptr<ExpressionGraph> graph,
          const std::string prefix,
          int dimInput,
          int dimState,
          Args... args)
    : CellType(graph, prefix, dimInput, dimState, args...)  {

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

  virtual RNNState applyState(std::vector<Expr> xWs,
                              RNNState state,
                              Expr mask = nullptr) {
    auto xWm = xWs.back();
    xWs.pop_back();

    auto sUm = affine(state.output, Um_, bm_);
    if(CellType::layerNorm_)
      sUm = layer_norm(sUm, gamma2m_);

    auto mstate = xWm * sUm;

    return CellType::applyState(xWs, RNNState({mstate, state.cell}), mask);
  }
};

typedef Multiplicative<FastLSTM> MLSTM;
typedef Multiplicative<GRU> MGRU;

/******************************************************************************/
// SlowLSTM and TestLSTM are for comparing efficient kernels for gradients with
// naive but correct LSTM version.

class SlowLSTM : public Cell {
private:
  std::string prefix_;

  Expr Uf_, Wf_, bf_;
  Expr Ui_, Wi_, bi_;
  Expr Uo_, Wo_, bo_;
  Expr Uc_, Wc_, bc_;

public:
  template <typename... Args>
  SlowLSTM(Ptr<ExpressionGraph> graph,
      const std::string prefix,
      int dimInput,
      int dimState,
      Args... args)
      : Cell(dimInput, dimState), prefix_{prefix} {

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

  RNNState apply(std::vector<Expr> inputs,
                 RNNState state,
                 Expr mask = nullptr) {
    return applyState(applyInput(inputs), state, mask);
  }

  std::vector<Expr> applyInput(std::vector<Expr> inputs) {
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

  RNNState applyState(std::vector<Expr> xWs,
                      RNNState state,
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

  size_t numStates() { return 2; }
};

/******************************************************************************/

class TestLSTM : public Cell {
private:
  std::string prefix_;

  Expr U_, W_, b_;

public:
  template <typename... Args>
  TestLSTM(Ptr<ExpressionGraph> graph,
      const std::string prefix,
      int dimInput,
      int dimState,
      Args... args)
      : Cell(dimInput, dimState), prefix_{prefix} {

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

  RNNState apply(std::vector<Expr> inputs,
                 RNNState state,
                 Expr mask = nullptr) {
    return applyState(applyInput(inputs), state, mask);
  }

  std::vector<Expr> applyInput(std::vector<Expr> inputs) {
    Expr input;
    if(inputs.size() > 1)
      input = concatenate(inputs, keywords::axis = 1);
    else
      input = inputs.front();

    auto xW = dot(input, W_);

    return {xW};
  }

  RNNState applyState(std::vector<Expr> xWs,
                      RNNState state,
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

  size_t numStates() { return 2; }
};

/******************************************************************************/

template <class Attention>
class AttentionCell : public Cell {
public:
  AttentionCell(int dimInput, int dimState)
    : Cell(dimInput, dimState) {}

  virtual Ptr<Attention> getAttention() = 0;
  virtual Expr getContexts() = 0;
  virtual Expr getLastContext() = 0;
};

template <class Cell1, class Attention, class Cell2>
class AttentionCellTmpl : public AttentionCell<Attention> {
private:
  Ptr<Cell> cell1_;
  Ptr<Cell> cell2_;
  Ptr<Attention> att_;

public:
  template <class... Args>
  AttentionCellTmpl(Ptr<ExpressionGraph> graph,
                    const std::string prefix,
                    int dimInput,
                    int dimState,
                    Ptr<Attention> att,
                    Args... args)
    : AttentionCell<Attention>(dimInput, dimState)
  {
    cell1_ = New<Cell1>(graph,
                        prefix + "_cell1",
                        dimInput,
                        dimState,
                        keywords::final = false,
                        args...);

    att_ = New<Attention>(att);

    cell2_ = New<Cell2>(graph,
                        prefix + "_cell2",
                        att_->outputDim(),
                        dimState,
                        keywords::final = true,
                        args...);
  }

  RNNState apply(std::vector<Expr> inputs,
                 RNNState state,
                 Expr mask = nullptr) {
    return applyState(applyInput(inputs), state, mask);
  }

  std::vector<Expr> applyInput(std::vector<Expr> inputs) {
    return cell1_->applyInput(inputs);
  }

  RNNState applyState(std::vector<Expr> xWs,
                      RNNState state,
                      Expr mask = nullptr) {
    if(cell1_->numStates() == cell2_->numStates()) {
      auto hidden = cell1_->applyState(xWs, state, mask);
      auto alignedSourceContext = att_->apply(hidden.output);
      return cell2_->apply({alignedSourceContext}, hidden, mask);
    }
    else if(cell1_->numStates() > cell2_->numStates()) {
      auto hidden = cell1_->applyState(xWs, state, mask);
      auto alignedSourceContext = att_->apply(hidden.output);
      auto output = cell2_->apply({alignedSourceContext}, hidden, mask);
      return { output.output, hidden.cell };
    }
    else {
      auto hidden = cell1_->applyState(xWs, state, mask);
      auto alignedSourceContext = att_->apply(hidden.output);
      return cell2_->apply({alignedSourceContext}, {hidden.output, state.cell}, mask);
    }
  }

  Ptr<Attention> getAttention() { return att_; }

  Expr getContexts() {
    return concatenate(att_->getContexts(), keywords::axis = 2);
  }

  Expr getLastContext() { return att_->getContexts().back(); }

  size_t numStates() { return cell1_->numStates(); }
};

typedef AttentionCellTmpl<GRU, GlobalAttention, GRU> CGRU;
typedef AttentionCellTmpl<MGRU, GlobalAttention, GRU> CMGRU;

typedef FastLSTM LSTM;

typedef AttentionCellTmpl<LSTM, GlobalAttention, LSTM> CLSTM;
typedef AttentionCellTmpl<MLSTM, GlobalAttention, LSTM> CMLSTM;
typedef AttentionCellTmpl<LSTM, GlobalAttention, GRU> CLSTMGRU;

class cell {
private:
  std::string type_;

public:
  cell(const std::string& type)
  : type_(type) {}

  template <typename ...Args>
  Ptr<Cell> operator()(Args&& ...args) {
    if(type_ == "gru")
      return New<GRU>(args...);
    if(type_ == "lstm")
      return New<LSTM>(args...);
    if(type_ == "mlstm")
      return New<MLSTM>(args...);
    if(type_ == "mgru")
      return New<MGRU>(args...);
    if(type_ == "tanh")
      return New<Tanh>(args...);
    return New<GRU>(args...);
  }
};

class att_cell {
private:
  std::string type_;

public:
  att_cell(const std::string& type)
  : type_(type) {}

  template <typename ...Args>
  Ptr<AttentionCell<GlobalAttention>> operator()(Args&& ...args) {
    if(type_ == "gru")
      return New<CGRU>(args...);
    if(type_ == "lstm")
      return New<CLSTM>(args...);
    if(type_ == "mgru")
      return New<CMGRU>(args...);
    if(type_ == "mlstm")
      return New<CMLSTM>(args...);
    if(type_ == "lstm-gru")
      return New<CLSTMGRU>(args...);
    return New<CGRU>(args...);
  }
};

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
