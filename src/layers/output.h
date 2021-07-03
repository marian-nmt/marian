#pragma once

#include "data/shortlist.h"
#include "generic.h"
#include "layers/factory.h"
#include "logits.h"
#include "marian.h"

namespace marian {

namespace mlp {

class Output : public LayerBase, public IUnaryLogitLayer, public IHasShortList {
private:
  // parameters held by this layer
  Expr Wt_;  // weight matrix is stored transposed for efficiency
  Expr b_;
  Expr lemmaEt_;  // re-embedding matrix for lemmas [lemmaDimEmb x lemmaVocabSize]
  bool isLegacyUntransposedW{false};  // legacy-model emulation: W is stored in non-transposed form
  bool hasBias_{true};

  Ptr<FactoredVocab> factoredVocab_;

  // optional parameters set/updated after construction
  Expr tiedParam_;
  Ptr<data::Shortlist> shortlist_;

  void lazyConstruct(int inputDim);

public:
  Output(Ptr<ExpressionGraph> graph, Ptr<Options> options)
      : LayerBase(graph, options), hasBias_{!options->get<bool>("output-omit-bias", false)} {
    clear();
  }

  void tieTransposed(Expr tied) {
    if(Wt_)
      ABORT_IF(tiedParam_.get() != tied.get(),
               "Tied output projection cannot be changed once weights have been created");
    else
      tiedParam_ = tied;
  }

  void setShortlist(Ptr<data::Shortlist> shortlist) override final {
    if(shortlist_)
      ABORT_IF(shortlist.get() != shortlist_.get(),
               "Output shortlist cannot be changed except after clear()");
    else {
      shortlist_ = shortlist;
    }
    // cachedShortWt_ and cachedShortb_ will be created lazily inside apply()
  }

  // this is expected to be called in sync with graph->clear(), which invalidates
  // cachedShortWt_ etc. in the graph's short-term cache
  void clear() override final {
    shortlist_ = nullptr;
  }

  Logits applyAsLogits(Expr input) override final;
};

}  // namespace mlp

}  // namespace marian
