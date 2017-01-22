#pragma once

#include "marian.h"
#include "graph/expression_graph.h"

namespace marian {

struct ParametersAttention {
  Expr Wa, ba, Ua, va;
};

class Attention {
  private:
    ParametersAttention params_;
    Expr context_;
    Expr softmaxMask_;
    Expr mappedContext_;
    std::vector<Expr> contexts_;

  public:
    Attention(ParametersAttention params,
              Expr context,
              Expr softmaxMask = nullptr)
     : params_(params),
       context_(context),
       softmaxMask_(nullptr) {

      mappedContext_ = dot(context_, params_.Ua);
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

      auto mappedState = dot(state, params_.Wa);
      auto temp = tanhPlus3(mappedState, mappedContext_ , params_.ba);

      // @TODO: horrible ->
      auto e = reshape(
        transpose(
          softmax(
            transpose(
              reshape(
                dot(temp, params_.va),
                {srcWords, dimBatch})),
            softmaxMask_)),
        {dimBatch, 1, srcWords});
      // <- horrible

      auto alignedSource = weighted_average(context_, e, axis=2);
      contexts_.push_back(alignedSource);
      return alignedSource;
    }

    std::vector<Expr>& getContexts() {
      return contexts_;
    }
};

}
