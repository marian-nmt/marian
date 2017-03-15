#pragma once

#include "marian.h"
#include "graph/expression_graph.h"
#include "layers/rnn.h"

namespace marian {

struct AttentionNodeOp : public NaryNodeOp {

  template <typename ...Args>
  AttentionNodeOp(const std::vector<Expr>& nodes, Args ...args)
    : NaryNodeOp(nodes,
                 keywords::shape=newShape(nodes)) {}

  Shape newShape(const std::vector<Expr>& nodes) {
    Shape shape = nodes[1]->shape();

    Shape vaShape =  nodes[0]->shape();
    Shape ctxShape = nodes[1]->shape();
    Shape stateShape = nodes[2]->shape();

    for(int i = 0; i < stateShape.size(); ++i) {
      UTIL_THROW_IF2(ctxShape[i] != stateShape[i] && ctxShape[i] != 1 && stateShape[i] != 1,
                     "Shapes cannot be broadcasted");
      shape.set(i, std::max(ctxShape[i], stateShape[i]));
    }

    UTIL_THROW_IF2(vaShape[0] != shape[1] || vaShape[1] != 1,
                   "Wrong size");

    shape.set(1, 1);
    return shape;
  }

  NodeOps forwardOps() {
    return {
      NodeOp(Att(val_,
                 children_[0]->val(),
                 children_[1]->val(),
                 children_[2]->val(),
                 children_.size() == 4 ? children_[3]->val() : nullptr))
    };
  }

  NodeOps backwardOps() {
    return {
      NodeOp(
        AttBack(
          children_[0]->grad(),
          children_[1]->grad(),
          children_[2]->grad(),
          children_.size() == 4 ? children_[3]->grad() : nullptr,
          children_[0]->val(),
          children_[1]->val(),
          children_[2]->val(),
          children_.size() == 4 ? children_[3]->val() : nullptr,
          adj_
        );
      )
    };
  }

  // do not check if node is trainable
  virtual void runBackward(const NodeOps& ops) {
    for(auto&& op : ops)
      op();
  }

  const std::string type() {
    return "Att-ops";
  }

  const std::string color() {
    return "yellow";
  }
};

Expr attOps(Expr va, Expr context, Expr state, Expr coverage=nullptr) {
  std::vector<Expr> nodes{va, context, state};
  if(coverage)
    nodes.push_back(coverage);

  int dimBatch = context->shape()[0];
  int dimWords = context->shape()[2];
  int dimBeam  = state->shape()[3];
  return reshape(Expression<AttentionNodeOp>(nodes),
                 {dimWords, dimBatch, 1, dimBeam});
}

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

    template <typename ...Args>
    GlobalAttention(const std::string prefix,
              Ptr<EncoderState> encState,
              int dimDecState,
              Args ...args)
     : encState_(encState),
       contextDropped_(encState->context),
       layerNorm_(Get(keywords::normalize, false, args...)),
       cov_(Get(keywords::coverage, nullptr, args...)) {

      int dimEncState = encState_->context->shape()[1];

      auto graph = encState_->context->graph();

      Wa_ = graph->param(prefix + "_W_comb_att", {dimDecState, dimEncState},
                         keywords::init=inits::glorot_uniform);
      Ua_ = graph->param(prefix + "_Wc_att", {dimEncState, dimEncState},
                         keywords::init=inits::glorot_uniform);
      va_ = graph->param(prefix + "_U_att", {dimEncState, 1},
                         keywords::init=inits::glorot_uniform);
      ba_ = graph->param(prefix + "_b_att", {1, dimEncState},
                         keywords::init=inits::zeros);

      dropout_ = Get(keywords::dropout_prob, 0.0f, args...);
      if(dropout_> 0.0f) {
        dropMaskContext_ = graph->dropout(dropout_, {1, dimEncState});
        dropMaskState_ = graph->dropout(dropout_, {1, dimDecState});
      }

      if(dropMaskContext_)
        contextDropped_ = dropout(contextDropped_, keywords::mask=dropMaskContext_);

      if(layerNorm_) {
        gammaContext_ = graph->param(prefix + "_att_gamma1", {1, dimEncState},
                                     keywords::init=inits::from_value(1.0));
        gammaState_ = graph->param(prefix + "_att_gamma2", {1, dimEncState},
                                   keywords::init=inits::from_value(1.0));

        mappedContext_ = layer_norm(dot(contextDropped_, Ua_), gammaContext_, ba_);
      }
      else {
        mappedContext_ = affine(contextDropped_, Ua_, ba_);
      }

      auto softmaxMask = encState_->mask;
      if(softmaxMask) {
        Shape shape = { softmaxMask->shape()[2],
                        softmaxMask->shape()[0] };
        softmaxMask_ = transpose(reshape(softmaxMask, shape));
      }
    }

    Expr apply(Expr state) {
      using namespace keywords;

      int dimBatch = contextDropped_->shape()[0];
      int srcWords = contextDropped_->shape()[2];
      int dimBeam  = state->shape()[3];

      if(dropMaskState_)
        state = dropout(state, keywords::mask=dropMaskState_);

      auto mappedState = dot(state, Wa_);
      if(layerNorm_)
        mappedState = layer_norm(mappedState, gammaState_);

      auto attReduce = attOps(va_, mappedContext_, mappedState);

      // @TODO: horrible ->
      auto e = reshape(transpose(softmax(transpose(attReduce), softmaxMask_)),
                       {dimBatch, 1, srcWords, dimBeam});
      // <- horrible

      auto alignedSource = weighted_average(encState_->context, e, axis=2);

      contexts_.push_back(alignedSource);
      alignments_.push_back(e);
      return alignedSource;
    }

    std::vector<Expr>& getContexts() {
      return contexts_;
    }

    int outputDim() {
      return encState_->context->shape()[1];
    }
};

}
