// Implementation of an IVocab that represents a factored representation.
// This is accessed via the IVocab interface for the base vocab functionality,
// and via dynamic_cast to FactoredVocab for factored-specific things used by
// the Embedding and Output layers.

#pragma once

#include "common/definitions.h"
#include "data/types.h"
#include "data/vocab_base.h"

#undef FACTOR_FULL_EXPANSION // define this to get full expansion. @TODO: infeasible for many factors; just delete this

namespace marian {

class FactoredVocab : public IVocab {
public:
  struct CSRData {
    Shape shape;
    std::vector<float> weights;
    std::vector<IndexType> indices;
    std::vector<IndexType> offsets;
  };

  // from IVocab:
  virtual size_t load(const std::string& factoredVocabPath, size_t maxSizeUnused = 0) override final;
  virtual void create(const std::string& vocabPath, const std::vector<std::string>& trainPaths, size_t maxSize) override final { vocabPath, trainPaths, maxSize; ABORT("Factored vocab cannot be created on the fly"); }
  virtual const std::string& canonicalExtension() const override final { return suffixes()[0]; }
  virtual const std::vector<std::string>& suffixes() const override final;
  virtual Word operator[](const std::string& word) const override final;
  virtual Words encode(const std::string& line, bool addEOS = true, bool inference = false) const override final;
  virtual std::string decode(const Words& sentence, bool ignoreEos = true) const override final;
  virtual std::string surfaceForm(const Words& sentence) const override final;
  virtual const std::string& operator[](Word id) const override final;
  virtual size_t size() const override final { return vocab_.size(); } // active factored vocabulary size (counting all valid combinations but not gaps)
  virtual std::string type() const override final { return "FactoredVocab"; }
  virtual Word getEosId() const override final { return eosId_; }
  virtual Word getUnkId() const override final { return unkId_; }
  virtual std::string toUpper(const std::string& line) const override final;
  virtual std::string toEnglishTitleCase(const std::string& line) const override final;
  virtual void transcodeToShortlistInPlace(WordIndex* ptr, size_t num) const override final;
  WordIndex getUnkIndex() const { return (WordIndex)getFactor(getUnkId(), 0); } // used in decoding
  virtual void createFake() override final { ABORT("[data] Fake FactoredVocab vocabulary not supported"); }
  virtual Word randWord() const override final;

  // factor-specific. These methods are consumed by Output and Embedding.
  size_t factorVocabSize() const { return factorVocab_.size(); } // total number of factors across all types
  size_t virtualVocabSize() const { return factorShape_.elements<size_t>(); } // valid WordIndex range (representing all factor combinations including gaps); virtual and huge
  virtual size_t lemmaSize() const override;

  CSRData csr_rows(const Words& words) const; // sparse matrix for summing up factors from the concatenated embedding matrix for each word
  void lemmaAndFactorsIndexes(const Words& words, std::vector<IndexType>& lemmaIndices, std::vector<float>& factorIndices) const;
#ifdef FACTOR_FULL_EXPANSION
  const CSRData& getGlobalFactorMatrix() const { return globalFactorMatrix_; }   // [v,u] (sparse) -> =1 if u is factor of v  --only used in getLogits()
#endif
  size_t getNumGroups() const { return groupRanges_.size(); }
  std::pair<size_t, size_t> getGroupRange(size_t g) const { return groupRanges_[g]; }   // [g] -> (u_begin, u_end)
  size_t getTotalFactorCount() const;
#ifdef FACTOR_FULL_EXPANSION
  const std::vector<float>& getGapLogMask() const { return gapLogMask_; } // [v] -inf if v is a gap entry, else 0
#endif

  // convert representations
  Word factors2word(const std::vector<size_t>& factors) const;
  void word2factors(Word word, std::vector<size_t>& factors) const;
  Word lemma2Word(size_t factor0Index) const;
  Word expandFactoredWord(Word word, size_t groupIndex, size_t factorIndex) const;
  bool canExpandFactoredWord(Word word, size_t groupIndex) const { return lemmaHasFactorGroup(getFactor(word, 0), groupIndex); }
  size_t getFactor(Word word, size_t groupIndex) const;
  bool lemmaHasFactorGroup(size_t factor0Index, size_t g) const { return lemmaHasFactorGroup_[factor0Index][g]; }
  const std::string& getFactorGroupPrefix(size_t groupIndex) const { return groupPrefixes_[groupIndex]; } // for diagnostics only
  const std::string& getFactorName(size_t groupIndex, size_t factorIndex) const { return factorVocab_[(WordIndex)(factorIndex + groupRanges_[groupIndex].first)]; }
  std::string decodeForDiagnostics(const Words& sentence) const;

  static constexpr size_t FACTOR_NOT_APPLICABLE = (SIZE_MAX - 1);
  static constexpr size_t FACTOR_NOT_SPECIFIED  = (SIZE_MAX - 2);
  static bool isFactorValid(size_t factorIndex) { return factorIndex < FACTOR_NOT_SPECIFIED; }

  static Ptr<FactoredVocab> tryCreateAndLoad(const std::string& path); // load from "vocab" option if it specifies a factored vocab
  std::string word2string(Word word) const;
  Word string2word(const std::string& w) const;
  bool tryGetFactor(const std::string& factorGroupName, size_t& groupIndex, size_t& factorIndex) const; // note: factorGroupName given without separator

private:
  void constructGroupInfoFromFactorVocab();
  void constructFactorIndexConversion();
  void rCompleteVocab(std::vector<size_t>& factorIndices, size_t g);
#ifdef FACTOR_FULL_EXPANSION
  void constructNormalizationInfoForVocab();
#endif
  size_t factorUnit2FactorIndex(WordIndex u) const;
private:
  // @TODO: Should we move WordLUT to utils?
  class WordLUT { // map between strings and WordIndex
    std::map<std::string, WordIndex> str2index_;
    std::map<WordIndex, std::string> index2str_;
  public:
    WordIndex add(const std::string& word, WordIndex index);
    const std::string& operator[](WordIndex index) const;
    WordIndex operator[](const std::string& word) const;
    bool contains(WordIndex index) const { return index2str_.find(index) != index2str_.end(); }
    bool tryFind(const std::string& word, WordIndex& index) const;
    size_t size() const { return str2index_.size(); }
    size_t load(const std::string& path);
    void dumpToFile(const std::string& path);
  };

  // main vocab
  Word eosId_{};
  Word unkId_{};
  WordLUT vocab_;
  size_t lemmaSize_;

  // factors
  char factorSeparator_ = '|';                         // separator symbol for parsing factored words
  WordLUT factorVocab_;                                // [factor name] -> factor index = row of E_
  std::vector<std::string> groupPrefixes_;             // [group id g] shared prefix of factors (used for grouping)
#ifdef FACTOR_FULL_EXPANSION
  CSRData globalFactorMatrix_;                         // [v,u] (sparse) -> =1 if u is factor of v
#endif
  std::vector<size_t> factorGroups_;                   // [u] -> group id of factor u
  std::vector<std::pair<size_t, size_t>> groupRanges_; // [group id g] -> (u_begin,u_end) index range of factors u for this group. These don't overlap.
  std::vector<std::vector<bool>> lemmaHasFactorGroup_; // [factor 0 index][g] -> true if lemma has factor group
  Shape factorShape_;                                  // [g] number of factors in each factor group
  std::vector<size_t> factorStrides_;                  // [g] stride for factor dimension
#ifdef FACTOR_FULL_EXPANSION
  std::vector<float> gapLogMask_;                      // [v] -1e8 if this is a gap, else 0
#endif
};

}  // namespace marian
