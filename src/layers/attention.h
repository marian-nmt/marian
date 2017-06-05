#pragma once

#include "common/definitions.h"
#include "graph/expression_graph.h"
#include "graph/expression_operators.h"
#include "models/states.h"

namespace marian {

Expr attOps(Expr va, Expr context, Expr state, Expr coverage = nullptr);

class GlobalAttention {
private:
  Expr Wa_, ba_, Ua_, va_;

  Expr gammaContext_, betaContext_;
  Expr gammaState_, betaState_;

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

  Expr cov_;

public:
  template <typename... Args>
  GlobalAttention(const std::string prefix,
                  Ptr<EncoderState> encState,
                  int dimDecState,
                  Args... args)
      : encState_(encState),
        contextDropped_(encState->getContext()),
        layerNorm_(Get(keywords::normalize, false, args...)),
        cov_(Get(keywords::coverage, nullptr, args...)) {
    int dimEncState = encState_->getContext()->shape()[1];

    auto graph = encState_->getContext()->graph();

    Wa_ = graph->param(prefix + "_W_comb_att",
                       {dimDecState, dimEncState},
                       keywords::init = inits::glorot_uniform);
    Ua_ = graph->param(prefix + "_Wc_att",
                       {dimEncState, dimEncState},
                       keywords::init = inits::glorot_uniform);
    va_ = graph->param(prefix + "_U_att",
                       {dimEncState, 1},
                       keywords::init = inits::glorot_uniform);
    ba_ = graph->param(
        prefix + "_b_att", {1, dimEncState}, keywords::init = inits::zeros);

    dropout_ = Get(keywords::dropout_prob, 0.0f, args...);
    if(dropout_ > 0.0f) {
      dropMaskContext_ = graph->dropout(dropout_, {1, dimEncState});
      dropMaskState_ = graph->dropout(dropout_, {1, dimDecState});
    }

    if(dropMaskContext_)
      contextDropped_
          = dropout(contextDropped_, keywords::mask = dropMaskContext_);

    if(layerNorm_) {
      gammaContext_ = graph->param(prefix + "_att_gamma1",
                                   {1, dimEncState},
                                   keywords::init = inits::from_value(1.0));
      gammaState_ = graph->param(prefix + "_att_gamma2",
                                 {1, dimEncState},
                                 keywords::init = inits::from_value(1.0));

      mappedContext_
          = layer_norm(dot(contextDropped_, Ua_), gammaContext_, ba_);
    } else {
      mappedContext_ = affine(contextDropped_, Ua_, ba_);
    }

    auto softmaxMask = encState_->getMask();
    if(softmaxMask) {
      Shape shape = {softmaxMask->shape()[2], softmaxMask->shape()[0]};
      softmaxMask_ = transpose(reshape(softmaxMask, shape));
    }
  }

  Expr apply(Expr state) {
    using namespace keywords;

    int dimBatch = contextDropped_->shape()[0];
    int srcWords = contextDropped_->shape()[2];
    int dimBeam = state->shape()[3];

    if(dropMaskState_)
      state = dropout(state, keywords::mask = dropMaskState_);

    auto mappedState = dot(state, Wa_);
    if(layerNorm_)
      mappedState = layer_norm(mappedState, gammaState_);

    auto attReduce = attOps(va_, mappedContext_, mappedState);

    // @TODO: horrible ->
    auto e = reshape(transpose(softmax(transpose(attReduce), softmaxMask_)),
                     {dimBatch, 1, srcWords, dimBeam});
    // <- horrible

    auto alignedSource = weighted_average(encState_->getContext(), e, axis = 2);

    contexts_.push_back(alignedSource);
    alignments_.push_back(e);
    return alignedSource;
  }

  std::vector<Expr>& getContexts() { return contexts_; }

  std::vector<Expr>& getAlignments() { return alignments_; }

  int outputDim() { return encState_->getContext()->shape()[1]; }
};
}
