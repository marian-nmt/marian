// Implementation of an IVocab that represents a factored representation.
// This is accessed via the IVocab interface, and also by Embedding and Output
// layers directly.

#pragma once

#include "common/definitions.h"
#include "data/types.h"
#include "data/vocab.h"
#include "data/vocab_base.h"

#include <numeric> // for std::iota()

namespace marian {

class FactoredVocab : public IVocab {
public:
  struct CSRData {
    Shape shape;
    std::vector<float> weights;
    std::vector<IndexType> indices;
    std::vector<IndexType> offsets;
  };

  FactoredVocab(Ptr<Options> options) : options_(options), factorVocab_(New<Options>(), 0) { }

  // from IVocab:
  virtual size_t load(const std::string& factoredVocabPath, size_t maxSizeUnused = 0) override final;

  virtual void create(const std::string& vocabPath, const std::vector<std::string>& trainPaths, size_t maxSize) override final { vocabPath, trainPaths, maxSize; ABORT("Factored vocab cannot be created on the fly"); }
  virtual const std::string& canonicalExtension() const override final { return suffixes()[0]; }
  virtual const std::vector<std::string>& suffixes() const override final { const static std::vector<std::string> exts{".fm"}; return exts; }
  virtual Word operator[](const std::string& word) const override final;
  virtual Words encode(const std::string& line, bool addEOS = true, bool inference = false) const override final;
  virtual std::string decode(const Words& sentence, bool ignoreEos = true) const override final;
  virtual const std::string& operator[](Word id) const override final;
  virtual size_t size() const override final;
  virtual std::string type() const override final { return "FactoredVocab"; }
  virtual Word getEosId() const override { return eosId_; }
  virtual Word getUnkId() const override { return unkId_; }
  virtual void createFake() override final;

  // factor-specific. These methods are consumed by Output and Embedding.
  size_t factorVocabSize() const { return factorVocab_.size(); }

  CSRData csr_rows(const std::vector<IndexType>& words) const;

  const CSRData& getGlobalFactorMatrix() const { return globalFactorMatrix_; }   // [v,u] (sparse) -> =1 if u is factor of v
  size_t getNumGroups() const { return groupRanges_.size(); }
  std::pair<size_t, size_t>     getGroupRange(size_t g)    const { return groupRanges_[g]; }   // [g] -> (u_begin, u_end)
  const std::vector<float>&     getFactorMasks(size_t g)   const { return factorMasks_[g]; }   // [g][v] 1.0 if word v has factor g
  const std::vector<IndexType>& getFactorIndices(size_t g) const { return factorIndices_[g]; } // [g][v] local index u_g = u - u_g,begin of factor g for word v; 0 if not a factor

private:
  Ptr<Options> options_;

  // main vocab
  Word eosId_{};
  Word unkId_{};

  // factors
  Vocab factorVocab_;                                  // [factor name] -> factor index = row of E_
  std::vector<std::vector<WordIndex>> factorMap_;      // [word index v] -> set of factor indices u
  std::vector<int> factorRefCounts_;                   // [factor index u] -> how often factor u is referenced in factorMap_
  CSRData globalFactorMatrix_;                         // [v,u] (sparse) -> =1 if u is factor of v
  std::vector<size_t> factorGroups_;                   // [u] -> group id of factor u
  std::vector<std::pair<size_t, size_t>> groupRanges_; // [group id g] -> (u_begin,u_end) index range of factors u for this group. These don't overlap.
  std::vector<std::vector<float>>     factorMasks_;    // [g][v] 1.0 if word v has factor g
  std::vector<std::vector<IndexType>> factorIndices_;  // [g][v] relative index u - u_begin of factor g (or any valid index if it does not have it; we use 0)
};

}  // namespace marian
