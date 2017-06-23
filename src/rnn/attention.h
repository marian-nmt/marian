#pragma once

#include "common/definitions.h"
#include "graph/expression_graph.h"
#include "graph/expression_operators.h"
#include "models/states.h"

#include "rnn/types.h"

namespace marian {

namespace rnn {

Expr attOps(Expr va, Expr context, Expr state, Expr coverage = nullptr);

class GlobalAttention : public CellInput {
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

public:
  GlobalAttention(Ptr<ExpressionGraph> graph,
                  Ptr<Options> options,
                  Ptr<EncoderState> encState)
      : CellInput(options),
        encState_(encState),
        contextDropped_(encState->getContext()) {

    int dimDecState = options_->get<int>("dimState");
    dropout_ = options_->get<float>("dropout");
    layerNorm_ = options_->get<bool>("layer-normalization");
    std::string prefix = options_->get<std::string>("prefix");

    int dimEncState = encState_->getContext()->shape()[1];

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

  Expr apply(State state) {
    using namespace keywords;
    auto recState = state.output;

    int dimBatch = contextDropped_->shape()[0];
    int srcWords = contextDropped_->shape()[2];
    int dimBeam = recState->shape()[3];


    if(dropMaskState_)
      recState = dropout(recState, keywords::mask = dropMaskState_);

    auto mappedState = dot(recState, Wa_);
    if(layerNorm_)
      mappedState = layer_norm(mappedState, gammaState_);

    auto attReduce = attOps(va_, mappedContext_, mappedState);

    // @TODO: horrible ->
    auto e = reshape(transpose(softmax(transpose(attReduce), softmaxMask_)),
                     {dimBatch, 1, srcWords, dimBeam});
    // <- horrible

    auto alignedSource = weighted_average(encState_->getAttended(), e, axis = 2);

    contexts_.push_back(alignedSource);
    alignments_.push_back(e);
    return alignedSource;
  }

  std::vector<Expr>& getContexts() { return contexts_; }

  Expr getContext() {
    return concatenate(contexts_, keywords::axis=2);
  }

  std::vector<Expr>& getAlignments() { return alignments_; }

  virtual void clear() {
    contexts_.clear();
    alignments_.clear();
  }

  int dimOutput() { return encState_->getContext()->shape()[1]; }
};

using Attention = GlobalAttention;

}

}
