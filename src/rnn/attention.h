#pragma once

#include "marian.h"
#include "models/states.h"
#include "rnn/types.h"

namespace marian {
namespace rnn {

Expr attOps(Expr va, Expr context, Expr state);

class GlobalAttention : public CellInput {
private:
  Expr Wa_, ba_, Ua_, va_;

  Expr gammaContext_;
  Expr gammaState_;

  Ptr<EncoderState> encState_;
  Expr softmaxMask_;
  Expr mappedContext_;
  std::vector<Expr> contexts_;
  std::vector<Expr> alignments_;
  bool layerNorm_;
  float dropout_;

  Expr contextDropped_;
  Expr dropMaskContext_;
  Expr dropMaskState_;

  // for Nematus-style layer normalization
  Expr Wc_att_lns_, Wc_att_lnb_;
  Expr W_comb_att_lns_, W_comb_att_lnb_;
  bool nematusNorm_;

public:
  GlobalAttention(Ptr<ExpressionGraph> graph,
                  Ptr<Options> options,
                  Ptr<EncoderState> encState)
      : CellInput(options),
        encState_(encState),
        contextDropped_(encState->getContext()) {
    int dimDecState = options_->get<int>("dimState");
    dropout_ = options_->get<float>("dropout", 0);
    layerNorm_ = options_->get<bool>("layer-normalization", false);
    nematusNorm_ = options_->get<bool>("nematus-normalization", false);
    std::string prefix = options_->get<std::string>("prefix");

    int dimEncState = encState_->getContext()->shape()[-1];

    Wa_ = graph->param(prefix + "_W_comb_att",
                       {dimDecState, dimEncState},
                       inits::glorot_uniform);
    Ua_ = graph->param(
        prefix + "_Wc_att", {dimEncState, dimEncState}, inits::glorot_uniform);
    va_ = graph->param(
        prefix + "_U_att", {dimEncState, 1}, inits::glorot_uniform);
    ba_ = graph->param(prefix + "_b_att", {1, dimEncState}, inits::zeros);

    if(dropout_ > 0.0f) {
      dropMaskContext_ = graph->dropout(dropout_, {1, dimEncState});
      dropMaskState_ = graph->dropout(dropout_, {1, dimDecState});
    }

    if(dropMaskContext_)
      contextDropped_ = dropout(contextDropped_, dropMaskContext_);

    if(layerNorm_) {
      if(nematusNorm_) {
        // instead of gammaContext_
        Wc_att_lns_ = graph->param(
            prefix + "_Wc_att_lns", {1, dimEncState}, inits::from_value(1.f));
        Wc_att_lnb_ = graph->param(
            prefix + "_Wc_att_lnb", {1, dimEncState}, inits::zeros);
        // instead of gammaState_
        W_comb_att_lns_ = graph->param(prefix + "_W_comb_att_lns",
                                       {1, dimEncState},
                                       inits::from_value(1.f));
        W_comb_att_lnb_ = graph->param(
            prefix + "_W_comb_att_lnb", {1, dimEncState}, inits::zeros);

        mappedContext_ = layerNorm(affine(contextDropped_, Ua_, ba_),
                                   Wc_att_lns_,
                                   Wc_att_lnb_,
                                   NEMATUS_LN_EPS);
      } else {
        gammaContext_ = graph->param(
            prefix + "_att_gamma1", {1, dimEncState}, inits::from_value(1.0));
        gammaState_ = graph->param(
            prefix + "_att_gamma2", {1, dimEncState}, inits::from_value(1.0));

        mappedContext_
            = layerNorm(dot(contextDropped_, Ua_), gammaContext_, ba_);
      }

    } else {
      mappedContext_ = affine(contextDropped_, Ua_, ba_);
    }

    auto softmaxMask = encState_->getMask();
    if(softmaxMask) {
      Shape shape = {softmaxMask->shape()[-3], softmaxMask->shape()[-2]};
      softmaxMask_ = transpose(reshape(softmaxMask, shape));
    }
  }

  Expr apply(State state) override {
    using namespace keywords;
    auto recState = state.output;

    int dimBatch = contextDropped_->shape()[-2];
    int srcWords = contextDropped_->shape()[-3];
    int dimBeam = 1;
    if(recState->shape().size() > 3)
      dimBeam = recState->shape()[-4];

    if(dropMaskState_)
      recState = dropout(recState, dropMaskState_);

    auto mappedState = dot(recState, Wa_);
    if(layerNorm_) {
      if(nematusNorm_) {
        mappedState = layerNorm(
            mappedState, W_comb_att_lns_, W_comb_att_lnb_, NEMATUS_LN_EPS);
      } else {
        mappedState = layerNorm(mappedState, gammaState_);
      }
    }

    auto attReduce = attOps(va_, mappedContext_, mappedState);

    // @TODO: horrible ->
    auto e = reshape(transpose(softmax(transpose(attReduce), softmaxMask_)),
                     {dimBeam, srcWords, dimBatch, 1});
    // <- horrible

    auto alignedSource = scalar_product(encState_->getAttended(), e, axis = -3);

    contexts_.push_back(alignedSource);
    alignments_.push_back(e);
    return alignedSource;
  }

  std::vector<Expr>& getContexts() { return contexts_; }

  Expr getContext() { return concatenate(contexts_, keywords::axis = -3); }

  std::vector<Expr>& getAlignments() { return alignments_; }

  virtual void clear() override {
    contexts_.clear();
    alignments_.clear();
  }

  int dimOutput() override { return encState_->getContext()->shape()[-1]; }
};

using Attention = GlobalAttention;
}  // namespace rnn
}  // namespace marian
