#include "marian.h"

#include "layers/generic.h"
#include "layers/constructors.h"
#include "layers/loss.h"
#include "data/factored_vocab.h"
#include "rnn/types.h"     // for State::select()
#include "models/states.h" // for EncoderState
#include "layers/lsh.h"

namespace marian {
  Logits::Logits(Expr logits) : Logits(New<RationalLoss>(logits, nullptr)) {} // single-output constructor from Expr only (RationalLoss has no count)

  Ptr<ExpressionGraph> Logits::graph() const {
    ABORT_IF(logits_.empty(), "Empty logits object??");
    return logits_.front()->loss()->graph();
  }

  // This function assumes that the object holds one or more factor logits.
  // It applies the supplied loss function to each, and then returns the aggregate loss over all factors.
  Expr Logits::applyLossFunction(const Words& labels, const std::function<Expr(Expr/*logits*/, Expr/*indices*/)>& lossFn) const {
    LOG_ONCE(info, "[logits] Applying loss function for {} factor(s)", logits_.size());
    ABORT_IF(empty(), "Attempted to read out logits on empty Logits object");

    auto firstLogits = logits_.front()->loss();
    ABORT_IF(labels.size() * firstLogits->shape()[-1] != firstLogits->shape().elements(),
             "Labels not matching logits shape ({} != {}, {})??",
             labels.size() * firstLogits->shape()[-1],
             firstLogits->shape().elements(),
             firstLogits->shape());

    // base case (no factors)
    if (!factoredVocab_) {
      ABORT_IF(logits_.size() != 1, "Factors without factor mappings??");
      return lossFn(firstLogits, indices(toWordIndexVector(labels)));
    }

    auto numGroups = factoredVocab_->getNumGroups();

    // split labels into individual factor labels
    auto allMaskedFactoredLabels = factorizeWords(labels); // [numGroups][labels.size()] = [numGroups][B... flattened]

    //Expr indices = this->indices(toWordIndexVector(labels));
    // accumulate all CEs for all words that have the factor
    // Memory-wise, this is cheap, all temp objects below are batches of scalars or lookup vectors.
    Expr loss;
    for (size_t g = 0; g < numGroups; g++) {
      if (!logits_[g])
        continue; // empty factor  --@TODO: use an array of indices of non-empty logits_[]
      const auto& maskedFactoredLabels = allMaskedFactoredLabels[g]; // array of (word index, mask)
      auto factorIndices = indices (maskedFactoredLabels.indices); // [B... flattened] factor-label indices, or 0 if factor does not apply
      auto factorMask    = constant(maskedFactoredLabels.masks);   // [B... flattened] loss values get multiplied with 0 for labels that don't have this factor
      auto factorLogits  = logits_[g];                             // [B... * Ug] label-wise loss values (not aggregated yet)
      // For each location in [B...] select [indices[B...]]. If not using factor, select [0] and mask it out next.
      auto factorLoss = lossFn(factorLogits->loss(), factorIndices); // [B... x 1]
      if(loss)
        factorLoss = cast(factorLoss, loss->value_type());
      factorLoss = factorLoss * cast(reshape(factorMask, factorLoss->shape()), factorLoss->value_type()); // mask out factor for words that do not have that factor
      loss = loss ? (loss + factorLoss) : factorLoss; // [B... x 1]
    }
    return loss;
  }

  // This function assumes this object holds a single factor that represents a rational loss (with count).
  //Ptr<RationalLoss> Logits::getRationalLoss() const {
  //  ABORT_IF(logits_.size() != 1 || factoredVocab_, "getRationalLoss() cannot be used on multi-factor outputs");
  //  ABORT_IF(!logits_.front()->count(), "getRationalLoss() used on rational loss without count");
  //  return logits_.front();
  //}

  // get logits for one factor group
  // For groupIndex == 0, the function also requires the shortlist if there is one.
  Expr Logits::getFactoredLogits(size_t groupIndex, Ptr<data::Shortlist> shortlist /*= nullptr*/, const std::vector<IndexType>& hypIndices /*= {}*/, size_t beamSize /*= 0*/) const {
    ABORT_IF(empty(), "Attempted to read out logits on empty Logits object");
    
    auto sel = logits_[groupIndex]->loss(); // [localBeamSize, 1, dimBatch, dimFactorVocab]

    // normalize for decoding:
    //  - all secondary factors: subtract their max
    //  - lemma: add all maxes of applicable factors
    if (groupIndex > 0) {
      sel = sel - max(sel, -1);
    }
    else {
      auto numGroups = getNumFactorGroups();
      for (size_t g = 1; g < numGroups; g++) {
        auto factorMaxima = max(logits_[g]->loss(), -1); // we cast since loss is likely ce-loss which has type float32
        auto factorMasks = constant(getFactorMasks(g, shortlist ? shortlist->indices() : std::vector<WordIndex>()));
        sel = sel + cast(factorMaxima, sel->value_type()) * cast(factorMasks, sel->value_type()); // those lemmas that don't have a factor get multiplied with 0
      }
    }

    // if selIdx are given, then we must reshuffle accordingly
    if (!hypIndices.empty()) // use the same function that shuffles decoder state
      sel = rnn::State::select(sel, hypIndices, (int)beamSize, /*isBatchMajor=*/false);
    
    return sel;
  }

  // used for breakDown() only
  // Index is flattened
  Tensor Logits::getFactoredLogitsTensor(size_t groupIndex) const {
    ABORT_IF(empty(), "Attempted to read out logits on empty Logits object");
    return logits_[groupIndex]->loss()->val();
  }

  // This function assumes that the object holds one or more factor logits, which are summed up
  // into output-vocab logits according to the factored model (with correct normalization of factors).
  // This is infeasible for realistic factor sets, and therefore only implemented for 1 factor.
  // @TODO: remove altogether
  Expr Logits::getLogits() const {
    ABORT_IF(empty(), "Attempted to read out logits on empty Logits object");
    if (!factoredVocab_) {
      ABORT_IF(logits_.size() != 1, "Factors without factor mappings??");
      return getFactoredLogits(0);
    }

#ifdef FACTOR_FULL_EXPANSION
    // compute normalized factor log probs
    std::vector<Expr> logProbs(logits_.size());
    for (size_t g = 0; g < logits_.size(); g++)
      logProbs[g] = logsoftmax(logits_[g]->loss());
    auto y = concatenate(logProbs, /*axis=*/ -1);

    // sum up the unit logits across factors for each target word
    auto graph = y->graph();
    auto factorMatrix = factoredVocab_->getGlobalFactorMatrix(); // [V x U]
    y = dot_csr(
        y,  // [B x U]
        factorMatrix.shape,
        graph->constant({(int)factorMatrix.weights.size()}, inits::fromVector(factorMatrix.weights)),
        graph->constant({(int)factorMatrix.indices.size()}, inits::fromVector(factorMatrix.indices), Type::uint32),
        graph->constant({(int)factorMatrix.offsets.size()}, inits::fromVector(factorMatrix.offsets), Type::uint32),
        /*transB=*/ true); // -> [B x V]

    // mask out gaps
    auto gapLogMask = factoredVocab_->getGapLogMask(); // [V]
    y = y + graph->constant({ (int)gapLogMask.size() }, inits::fromVector(gapLogMask));

    return y;
#else
    ABORT("getLogits() no longer supported for actual factored vocab"); // because it is infeasible
#endif
  }


}  // namespace marian
