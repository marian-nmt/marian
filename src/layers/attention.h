#pragma once

#include "marian.h"
#include "graph/expression_graph.h"

namespace marian {

class Attention {
  private:
    Expr Wa_, ba_, Ua_, va_;

    Expr context_;
    Expr softmaxMask_;
    Expr mappedContext_;
    std::vector<Expr> contexts_;

  public:

    template <typename ...Args>
    Attention(const std::string prefix,
              Expr context,
              int dimDecState,
              Args ...args)
     : context_(context),
       softmaxMask_(nullptr) {

      int dimEncState = context->shape()[1];
      auto graph = context->graph();

      Wa_ = graph->param(prefix + "_W_comb_att", {dimDecState, dimEncState},
                         keywords::init=inits::glorot_uniform);
      ba_ = graph->param(prefix + "_b_att", {1, dimEncState},
                         keywords::init=inits::zeros);
      Ua_ = graph->param(prefix + "_Wc_att", {dimEncState, dimEncState},
                         keywords::init=inits::glorot_uniform);
      va_ = graph->param(prefix + "_U_att", {dimEncState, 1},
                         keywords::init=inits::glorot_uniform);

      mappedContext_ = affine(context_, Ua_, ba_);

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

      // do this in single reduction
      auto temp = tanh(mappedState, mappedContext_);
      auto temp2 = reshape(dot(temp, va_),
                           {srcWords, dimBatch});

      // @TODO: horrible ->
      auto e = reshape(
        transpose(softmax(transpose(temp2), softmaxMask_)),
        {dimBatch, 1, srcWords});
      // <- horrible

      auto alignedSource = weighted_average(context_, e, axis=2);

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
