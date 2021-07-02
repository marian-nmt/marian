#pragma once

#include "data/shortlist.h"
#include "generic.h"
#include "marian.h"

namespace marian {

class FactoredVocab;

// To support factors, any output projection (that is followed by a softmax) must
// retain multiple outputs, one for each factor. Such layer returns not a single Expr,
// but a Logits object that contains multiple.
// This allows to compute softmax values in a factored manner, where we never create
// a fully expanded list of all factor combinations.
class RationalLoss;
class Logits {
public:
  Logits() {}
  explicit Logits(Ptr<RationalLoss> logits);  // single-output constructor
  explicit Logits(Expr logits);  // single-output constructor from Expr only (RationalLoss has no count)
  Logits(std::vector<Ptr<RationalLoss>>&& logits,
         Ptr<FactoredVocab> embeddingFactorMapping);  // factored-output constructor

  Expr getLogits() const;  // assume it holds logits: get them, possibly aggregating over factors
  Expr getFactoredLogits(
      size_t groupIndex,
      Ptr<data::Shortlist> shortlist = nullptr,
      const std::vector<IndexType>& hypIndices = {},
      size_t beamSize = 0) const;  // get logits for only one factor group, with optional reshuffle
  // Ptr<RationalLoss> getRationalLoss() const; // assume it holds a loss: get that
  Expr applyLossFunction(
      const Words& labels,
      const std::function<Expr(Expr /*logits*/, Expr /*indices*/)>& lossFn) const;
  Logits applyUnaryFunction(
      const std::function<Expr(Expr)>& f) const;  // clone this but apply f to all loss values
  Logits applyUnaryFunctions(const std::function<Expr(Expr)>& f1,
                             const std::function<Expr(Expr)>& fother)
      const;  // clone this but apply f1 to first and fother to to all other values

  struct MaskedFactorIndices {
    std::vector<WordIndex> indices;  // factor index, or 0 if masked
    std::vector<float> masks;
    void reserve(size_t n) {
      indices.reserve(n);
      masks.reserve(n);
    }
    void push_back(size_t factorIndex);  // push back into both arrays, setting mask and index to 0
                                         // for invalid entries
    MaskedFactorIndices() {}
    MaskedFactorIndices(const Words& words) {
      indices = toWordIndexVector(words);
    }  // we can leave masks uninitialized for this special use case
  };
  std::vector<MaskedFactorIndices> factorizeWords(
      const Words& words) const;  // breaks encoded Word into individual factor indices
  Tensor getFactoredLogitsTensor(size_t factorGroup) const;  // used for breakDown() only
  size_t getNumFactorGroups() const { return logits_.size(); }
  bool empty() const { return logits_.empty(); }
  Logits withCounts(
      const Expr& count) const;  // create new Logits with 'count' implanted into all logits_
private:
  // helper functions
  Ptr<ExpressionGraph> graph() const;
  Expr constant(const Shape& shape, const std::vector<float>& data) const {
    return graph()->constant(shape, inits::fromVector(data));
  }
  Expr constant(const Shape& shape, const std::vector<uint32_t>& data) const {
    return graph()->constant(shape, inits::fromVector(data));
  }
  template <typename T>
  Expr constant(const std::vector<T>& data) const {
    return constant(Shape{(int)data.size()}, data);
  }  // same as constant() but assuming vector
  Expr indices(const std::vector<uint32_t>& data) const {
    return graph()->indices(data);
  }  // actually the same as constant(data) for this data type
  std::vector<float> getFactorMasks(size_t factorGroup,
                                    const std::vector<WordIndex>& indices) const;
  std::vector<float> getFactorMasks(size_t factorGroup, Expr indicesExpr) const; // same as above but separate indices for each batch and beam 

private:
  // members
  // @TODO: we don't use the RationalLoss component anymore, can be removed again, and replaced just
  // by the Expr
  std::vector<Ptr<RationalLoss>> logits_;  // [group id][B..., num factors in group]
  Ptr<FactoredVocab> factoredVocab_;
};

// Unary function that returns a Logits object
// Also implements IUnaryLayer, since Logits can be cast to Expr.
// This interface is implemented by all layers that are of the form of a unary function
// that returns multiple logits, to support factors.
struct IUnaryLogitLayer : public IUnaryLayer {
  virtual Logits applyAsLogits(Expr) = 0;
  virtual Logits applyAsLogits(const std::vector<Expr>& es) {
    ABORT_IF(es.size() > 1, "Not implemented");  // simple stub
    return applyAsLogits(es.front());
  }
  virtual Expr apply(Expr e) override { return applyAsLogits(e).getLogits(); }
  virtual Expr apply(const std::vector<Expr>& es) override { return applyAsLogits(es).getLogits(); }
};

}  // namespace marian
