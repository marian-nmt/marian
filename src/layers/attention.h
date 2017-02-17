#pragma once

#include "marian.h"
#include "graph/expression_graph.h"

namespace marian {

struct AttentionNodeOp : public NaryNodeOp {

  template <typename ...Args>
  AttentionNodeOp(const std::vector<Expr>& nodes, Args ...args)
    : NaryNodeOp(nodes,
                 keywords::shape=newShape(nodes)) {}

  Shape newShape(const std::vector<Expr>& nodes) {
    Shape shape = nodes[0]->shape();
    Shape shape2 = nodes[1]->shape();
    Shape shape3 = nodes[2]->shape();

    for(int i = 0; i < shape2.size(); ++i) {
      UTIL_THROW_IF2(shape[i] != shape2[i] && shape[i] != 1 && shape2[i] != 1,
                     "Shapes cannot be broadcasted");
      shape.set(i, std::max(shape[i], shape2[i]));
    }

    UTIL_THROW_IF2(shape3[0] != shape[1] || shape3[1] != 1,
                   "Wrong size");

    shape.set(1, 1);
    return shape;
  }

  NodeOps forwardOps() {
    return {
      NodeOp(Att(val_,
                 children_[0]->val(),
                 children_[1]->val(),
                 children_[2]->val()))
    };
  }

  NodeOps backwardOps() {
    return {
      NodeOp(
        AttBack(
          children_[0]->grad(),
          children_[1]->grad(),
          children_[2]->grad(),
          children_[0]->val(),
          children_[1]->val(),
          children_[2]->val(),
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

Expr attOps(Expr context, Expr state, Expr va) {
  std::vector<Expr> nodes{context, state, va};
  int dimBatch = context->shape()[0];
  int dimWords = context->shape()[2];
  return reshape(Expression<AttentionNodeOp>(nodes),
                 {dimWords, dimBatch});
}

class GlobalAttention {
  private:
    Expr Wa_, ba_, Ua_, va_;

    Expr gammaContext_, betaContext_;
    Expr gammaState_, betaState_;

    Expr context_;
    Expr softmaxMask_;
    Expr mappedContext_;
    std::vector<Expr> contexts_;
    bool layerNorm_;

  public:

    template <typename ...Args>
    GlobalAttention(const std::string prefix,
              Expr context,
              int dimDecState,
              Args ...args)
     : context_(context),
       softmaxMask_(nullptr),
       layerNorm_(Get(keywords::normalize, false, args...)) {

      int dimEncState = context->shape()[1];
      auto graph = context->graph();

      Wa_ = graph->param(prefix + "_W_comb_att", {dimDecState, dimEncState},
                         keywords::init=inits::glorot_uniform);
      Ua_ = graph->param(prefix + "_Wc_att", {dimEncState, dimEncState},
                         keywords::init=inits::glorot_uniform);
      va_ = graph->param(prefix + "_U_att", {dimEncState, 1},
                         keywords::init=inits::glorot_uniform);
      ba_ = graph->param(prefix + "_b_att", {1, dimEncState},
                         keywords::init=inits::zeros);

      if(layerNorm_) {
        gammaContext_ = graph->param(prefix + "_att_gamma1", {1, dimEncState},
                                     keywords::init=inits::from_value(1.0));
        gammaState_ = graph->param(prefix + "_att_gamma2", {1, dimEncState},
                                   keywords::init=inits::from_value(1.0));

        mappedContext_ = layer_norm(dot(context_, Ua_), gammaContext_, ba_);
      }
      else {
        mappedContext_ = affine(context_, Ua_, ba_);
      }

      auto softmaxMask = Get(keywords::mask, nullptr, args...);
      if(softmaxMask) {
        Shape shape = { softmaxMask->shape()[2],
                        softmaxMask->shape()[0] };
        softmaxMask_ = transpose(reshape(softmaxMask, shape));
      }
    }

    Expr apply(Expr state) {
      using namespace keywords;

      int dimBatch = context_->shape()[0];
      int srcWords = context_->shape()[2];

      auto mappedState = dot(state, Wa_);
      if(layerNorm_)
        mappedState = layer_norm(mappedState, gammaState_);

      auto attReduce = attOps(mappedContext_, mappedState, va_);

      // @TODO: horrible ->
      auto e = reshape(
        transpose(softmax(transpose(attReduce),
                          softmaxMask_)),
        {dimBatch, 1, srcWords});
      // <- horrible

      auto alignedSource = weighted_average(context_, e,
                                            axis=2);

      contexts_.push_back(alignedSource);
      return alignedSource;
    }

    std::vector<Expr>& getContexts() {
      return contexts_;
    }

    int outputDim() {
      return context_->shape()[1];
    }
};

}
