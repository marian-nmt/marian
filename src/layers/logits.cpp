#include "logits.h"
#include "data/factored_vocab.h"
#include "loss.h"
#include "rnn/types.h"  // for State::select()

namespace marian {
Logits::Logits(Expr logits)
    : Logits(New<RationalLoss>(logits, nullptr)) {
}  // single-output constructor from Expr only (RationalLoss has no count)

Logits::Logits(Ptr<RationalLoss> logits) {  // single-output constructor
  logits_.push_back(logits);
}

Logits::Logits(std::vector<Ptr<RationalLoss>>&& logits,
        Ptr<FactoredVocab> embeddingFactorMapping)  // factored-output constructor
    : logits_(std::move(logits)), factoredVocab_(embeddingFactorMapping) {
}

Ptr<ExpressionGraph> Logits::graph() const {
  ABORT_IF(logits_.empty(), "Empty logits object??");
  return logits_.front()->loss()->graph();
}

// This function assumes that the object holds one or more factor logits.
// It applies the supplied loss function to each, and then returns the aggregate loss over all
// factors.
Expr Logits::applyLossFunction(
    const Words& labels,
    const std::function<Expr(Expr /*logits*/, Expr /*indices*/)>& lossFn) const {
  LOG_ONCE(info, "[logits] Applying loss function for {} factor(s)", logits_.size());
  ABORT_IF(empty(), "Attempted to read out logits on empty Logits object");

  auto firstLogits = logits_.front()->loss();
  ABORT_IF(labels.size() * firstLogits->shape()[-1] != firstLogits->shape().elements(),
           "Labels not matching logits shape ({} != {}, {})??",
           labels.size() * firstLogits->shape()[-1],
           firstLogits->shape().elements(),
           firstLogits->shape());

  // base case (no factors)
  if(!factoredVocab_) {
    ABORT_IF(logits_.size() != 1, "Factors without factor mappings??");
    return lossFn(firstLogits, indices(toWordIndexVector(labels)));
  }

  auto numGroups = factoredVocab_->getNumGroups();

  // split labels into individual factor labels
  auto allMaskedFactoredLabels
      = factorizeWords(labels);  // [numGroups][labels.size()] = [numGroups][B... flattened]

  // Expr indices = this->indices(toWordIndexVector(labels));
  // accumulate all CEs for all words that have the factor
  // Memory-wise, this is cheap, all temp objects below are batches of scalars or lookup vectors.
  Expr loss;
  for(size_t g = 0; g < numGroups; g++) {
    if(!logits_[g])
      continue;  // empty factor  --@TODO: use an array of indices of non-empty logits_[]
    // clang-format off
    const auto& maskedFactoredLabels = allMaskedFactoredLabels[g];    // array of (word index, mask)
    auto factorIndices = indices(maskedFactoredLabels.indices);       // [B... flattened] factor-label indices, or 0 if factor does not apply
    auto factorMask    = constant(maskedFactoredLabels.masks);        // [B... flattened] loss values get multiplied with 0 for labels that don't have this factor
    auto factorLogits  = logits_[g];                                  // [B... * Ug] label-wise loss values (not aggregated yet)
    //std::cerr << "g=" << g << " factorLogits->loss()=" << factorLogits->loss()->shape() << std::endl;
    // For each location in [B...] select [indices[B...]]. If not using factor, select [0] and mask it out next.
    auto factorLoss    = lossFn(factorLogits->loss(), factorIndices); // [B... x 1]
    // clang-format on
    if(loss)
      factorLoss = cast(factorLoss, loss->value_type());
    factorLoss
        = factorLoss
          * cast(
              reshape(factorMask, factorLoss->shape()),
              factorLoss->value_type());  // mask out factor for words that do not have that factor
    loss = loss ? (loss + factorLoss) : factorLoss;  // [B... x 1]
  }
  return loss;
}

// This function assumes this object holds a single factor that represents a rational loss (with
// count).
// Ptr<RationalLoss> Logits::getRationalLoss() const {
//  ABORT_IF(logits_.size() != 1 || factoredVocab_, "getRationalLoss() cannot be used on
//  multi-factor outputs"); ABORT_IF(!logits_.front()->count(), "getRationalLoss() used on rational
//  loss without count"); return logits_.front();
//}

// get logits for one factor group
// For groupIndex == 0, the function also requires the shortlist if there is one.
Expr Logits::getFactoredLogits(size_t groupIndex,
                               Ptr<data::Shortlist> shortlist /*= nullptr*/,
                               const std::vector<IndexType>& hypIndices /*= {}*/,
                               size_t beamSize /*= 0*/) const {
  ABORT_IF(empty(), "Attempted to read out logits on empty Logits object");

  auto sel = logits_[groupIndex]->loss();  // [localBeamSize, 1, dimBatch, dimFactorVocab]

  // normalize for decoding:
  //  - all secondary factors: subtract their max
  //  - lemma: add all maxes of applicable factors
  if(groupIndex > 0) {
    sel = sel - max(sel, -1);
  } else {
    auto numGroups = getNumFactorGroups();
    for(size_t g = 1; g < numGroups; g++) {
      auto factorMaxima = max(logits_[g]->loss(),
                              -1);  // we cast since loss is likely ce-loss which has type float32
      Expr factorMasks;
      if (!shortlist) {
        factorMasks = constant(getFactorMasks(g, std::vector<WordIndex>()));
      }
      else {
        auto forward = [this, g](Expr out, const std::vector<Expr>& inputs) {
          Expr lastIndices = inputs[0];
          std::vector<float> masks = getFactorMasks(g, lastIndices);
          out->val()->set(masks);
        };

        //int currBeamSize = sel->shape()[0];
        //int batchSize = sel->shape()[2];
        Expr lastIndices = shortlist->getIndicesExpr();
        //assert(lastIndices->shape()[0] == currBeamSize || lastIndices->shape()[0] == 1);
        //assert(lastIndices->shape()[1] == batchSize || lastIndices->shape()[1] == 1);
        
        factorMasks = lambda({lastIndices}, lastIndices->shape(), Type::float32, forward);  
        
        const Shape &s = factorMasks->shape();
        factorMasks = reshape(factorMasks, {s[0], 1, s[1], s[2]});
      }
      factorMaxima = cast(factorMaxima, sel->value_type());
      factorMasks = cast(factorMasks, sel->value_type());

      Expr tmp = factorMaxima * factorMasks;
      sel = sel + tmp;  // those lemmas that don't have a factor
    }
  }

  // if selIdx are given, then we must reshuffle accordingly
  if(!hypIndices.empty())  // use the same function that shuffles decoder state
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
  if(!factoredVocab_) {
    ABORT_IF(logits_.size() != 1, "Factors without factor mappings??");
    return getFactoredLogits(0);
  }

#ifdef FACTOR_FULL_EXPANSION
  // compute normalized factor log probs
  std::vector<Expr> logProbs(logits_.size());
  for(size_t g = 0; g < logits_.size(); g++)
    logProbs[g] = logsoftmax(logits_[g]->loss());
  auto y = concatenate(logProbs, /*axis=*/-1);

  // clang-format off
  // sum up the unit logits across factors for each target word
  auto graph = y->graph();
  auto factorMatrix = factoredVocab_->getGlobalFactorMatrix();  // [V x U]
  y = dot_csr(
      y,  // [B x U]
      factorMatrix.shape,
      graph->constant({(int)factorMatrix.weights.size()}, inits::fromVector(factorMatrix.weights)),
      graph->constant({(int)factorMatrix.indices.size()}, inits::fromVector(factorMatrix.indices), Type::uint32),
      graph->constant({(int)factorMatrix.offsets.size()}, inits::fromVector(factorMatrix.offsets), Type::uint32),
      /*transB=*/true);  // -> [B x V]
  // clang-format on

  // mask out gaps
  auto gapLogMask = factoredVocab_->getGapLogMask();  // [V]
  y = y + graph->constant({(int)gapLogMask.size()}, inits::fromVector(gapLogMask));

  return y;
#else
  ABORT("getLogits() no longer supported for actual factored vocab");  // because it is infeasible
#endif
}

void Logits::MaskedFactorIndices::push_back(size_t factorIndex) {
  bool isValid = FactoredVocab::isFactorValid(factorIndex);
  indices.push_back(isValid ? (WordIndex)factorIndex : 0);
  masks.push_back((float)isValid);
}

std::vector<Logits::MaskedFactorIndices> Logits::factorizeWords(const Words& words)
    const {  // [numGroups][words.size()] -> breaks encoded Word into individual factor indices
  if(!factoredVocab_) {
    ABORT_IF(logits_.size() != 1, "Factors without factor mappings??");
    return {MaskedFactorIndices(words)};
  }
  auto numGroups = factoredVocab_->getNumGroups();
  std::vector<MaskedFactorIndices> res(numGroups);
  for(size_t g = 0; g < numGroups; g++) {
    auto& resg = res[g];
    resg.reserve(words.size());
    for(const auto& word : words)
      resg.push_back(factoredVocab_->getFactor(word, g));
  }
  return res;
}

//// use first factor of each word to determine whether it has a specific factor
// std::vector<float> Logits::getFactorMasks(const Words& words, size_t factorGroup) const { // 1.0
// for words that do have this factor; else 0
//  std::vector<float> res;
//  res.reserve(words.size());
//  for (const auto& word : words) {
//    auto lemma = factoredVocab_->getFactor(word, 0);
//    res.push_back((float)factoredVocab_->lemmaHasFactorGroup(lemma, factorGroup));
//  }
//  return res;
//}

// return a vector of 1 or 0 indicating for each lemma whether it has a specific factor
// If 'indices' is given, then return the masks for the indices; otherwise for all lemmas
std::vector<float> Logits::getFactorMasks(size_t factorGroup, const std::vector<WordIndex>& indices)
    const {  // [lemmaIndex] -> 1.0 for words that do have this factor; else 0
  size_t n
      = indices.empty()
            ? (factoredVocab_->getGroupRange(0).second - factoredVocab_->getGroupRange(0).first)
            : indices.size();
  std::vector<float> res;
  res.reserve(n);
  // @TODO: we should rearrange lemmaHasFactorGroup as vector[groups[i] of float; then move this
  // into FactoredVocab
  for(size_t i = 0; i < n; i++) {
    auto lemma = indices.empty() ? i : (indices[i] - factoredVocab_->getGroupRange(0).first);
    res.push_back((float)factoredVocab_->lemmaHasFactorGroup(lemma, factorGroup));
  }
  return res;
}

std::vector<float> Logits::getFactorMasks(size_t factorGroup, Expr indicesExpr)
    const {  // [lemmaIndex] -> 1.0 for words that do have this factor; else 0
  int batchSize = indicesExpr->shape()[0];
  int currBeamSize = indicesExpr->shape()[1];
  int numHypos = batchSize * currBeamSize;
  std::vector<WordIndex> indices;
  indicesExpr->val()->get(indices);

  //std::cerr << "indices=" << indices.size() << std::endl;
  size_t n
      = indices.empty()
            ? (factoredVocab_->getGroupRange(0).second - factoredVocab_->getGroupRange(0).first)
            : indices.size() / numHypos;
  std::vector<float> res;
  res.reserve(numHypos * n);
  //std::cerr << "n=" << n << std::endl;

  // @TODO: we should rearrange lemmaHasFactorGroup as vector[groups[i] of float; then move this
  // into FactoredVocab
  for (size_t hypoIdx = 0; hypoIdx < numHypos; ++hypoIdx) {
    for(size_t i = 0; i < n; i++) {
      size_t idx = hypoIdx * n + i;
      size_t lemma = indices.empty() ? i : (indices[idx] - factoredVocab_->getGroupRange(0).first);
      res.push_back((float)factoredVocab_->lemmaHasFactorGroup(lemma, factorGroup));
    }
  }
  return res;
}

Logits Logits::applyUnaryFunction(
    const std::function<Expr(Expr)>& f) const {  // clone this but apply f to all loss values
  std::vector<Ptr<RationalLoss>> newLogits;
  for(const auto& l : logits_)
    newLogits.emplace_back(New<RationalLoss>(f(l->loss()), l->count()));
  return Logits(std::move(newLogits), factoredVocab_);
}

Logits Logits::applyUnaryFunctions(const std::function<Expr(Expr)>& f1,
                                   const std::function<Expr(Expr)>& fother) const {
  std::vector<Ptr<RationalLoss>> newLogits;
  bool first = true;
  for(const auto& l : logits_) {
    newLogits.emplace_back(New<RationalLoss>((first ? f1 : fother)(l->loss()),
                                             l->count()));  // f1 for first, fother for all others
    first = false;
  }
  return Logits(std::move(newLogits), factoredVocab_);
}

// @TODO: code dup with above; we can merge it into applyToRationalLoss()
Logits Logits::withCounts(
    const Expr& count) const {  // create new Logits with 'count' implanted into all logits_
  std::vector<Ptr<RationalLoss>> newLogits;
  for(const auto& l : logits_)
    newLogits.emplace_back(New<RationalLoss>(l->loss(), count));
  return Logits(std::move(newLogits), factoredVocab_);
}
}  // namespace marian
