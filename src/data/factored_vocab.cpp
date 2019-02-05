#include "data/vocab_base.h"
#include "common/definitions.h"
#include "data/types.h"
#include "common/options.h"
#include "common/regex.h"
#include "data/factored_vocab.h"

namespace marian {

// mapPath = path to file with entries in order of vocab entries of the form
//   WORD FACTOR1 FACTOR2 FACTOR3...
// listPath = path to file that lists all FACTOR names
// vocab = original vocabulary
// Note: The WORD field in the map file is redundant. It is required for consistency checking only.
// Factors are grouped
//  - user specifies list-factor prefixes; all factors beginning with that prefix are in the same group
//  - factors within a group as multi-class and normalized that way
//  - groups of size 1 are interpreted as sigmoids, multiply with P(u) / P(u-1)
//  - one prefix must not contain another
//  - all factors not matching a prefix get lumped into yet another class (the lemmas)
//  - factor vocab must be sorted such that all groups are consecutive
//  - result of Output layer is nevertheless logits, not a normalized probability, due to the sigmoid entries
/*virtual*/ size_t FactoredVocab::load(const std::string& factoredVocabPath, size_t maxSizeUnused /*= 0*/) /*override final*/ {
  auto mapPath = factoredVocabPath;
  auto factorVocabPath = mapPath;
  factorVocabPath.back() = 'l'; // map .fm to .fl
  auto vocabPath = factorVocabPath;
  vocabPath[vocabPath.size() - 2] = 'w'; // map .fl to .wl  --@TODO: This should go away; esp. to allow per-stream vocabs

  // Note: We misuse the Vocab class a little.
  // Specifically, it means that the factorVocab_ must contain </s> and "<unk>".
  vocab_.load(vocabPath);
  auto vocabSize = vocab_.size();
  factorVocab_.load(factorVocabPath);
  auto numFactors = factorVocab_.size();

  // load and parse factorMap
  factorMap_.resize(vocabSize);
  factorRefCounts_.resize(numFactors);
  std::vector<std::string> tokens;
  io::InputFileStream in(mapPath);
  std::string line;
  size_t numTotalFactors = 0;
  for (WordIndex v = 0; io::getline(in, line); v++) {
    tokens.clear(); // @BUGBUG: should be done in split()
    utils::splitAny(line, tokens, " \t");
    ABORT_IF(tokens.size() < 2 || tokens.front() != vocab_[Word::fromWordIndex(v)], "Factor map must list words in same order as vocab, and have at least one factor per word", mapPath);
    for (size_t i = 1; i < tokens.size(); i++) {
      auto u = factorVocab_[tokens[i]].toWordIndex();
      factorMap_[v].push_back(u);
      factorRefCounts_[u]++;
    }
    numTotalFactors += tokens.size() - 1;
  }
  LOG(info, "[embedding] Factored-embedding map read with total/unique of {}/{} factors for {} words", numTotalFactors, numFactors, vocabSize);

  // form groups
  // @TODO: hard-coded for these initial experiments
  std::vector<std::string> groupPrefixes = {
    "@C",
    "@GL", "@GR"
  };
  groupPrefixes.insert(groupPrefixes.begin(), "(unassigned)");     // first group is fallback for normal words (the string is only used for messages)
  size_t numGroups = groupPrefixes.size();
  factorGroups_.resize(numFactors, 0);
  for (size_t g = 1; g < groupPrefixes.size(); g++) { // set group labels; what does not match any prefix will stay in group 0
    const auto& groupPrefix = groupPrefixes[g];
    for (WordIndex u = 0; u < numFactors; u++)
      if (utils::beginsWith(factorVocab_[Word::fromWordIndex(u)], groupPrefix)) {
        ABORT_IF(factorGroups_[u] != 0, "Factor {} matches multiple groups, incl. {}", factorVocab_[Word::fromWordIndex(u)], groupPrefix);
        factorGroups_[u] = g;
      }
  }
  // determine group index ranges
  groupRanges_.resize(numGroups, { SIZE_MAX, (size_t)0 });
  std::vector<size_t> groupCounts(numGroups); // number of group members
  for (WordIndex u = 0; u < numFactors; u++) { // determine ranges; these must be non-overlapping, verified via groupCounts
    auto g = factorGroups_[u];
    if (groupRanges_[g].first > u)
        groupRanges_[g].first = u;
    if (groupRanges_[g].second < u + 1)
        groupRanges_[g].second = u + 1;
    groupCounts[g]++;
  }
  // create mappings needed for normalization in factored outputs
  factorMasks_  .resize(numGroups, std::vector<float>(vocabSize, 0));     // [g][v] 1.0 if word v has factor g
  factorIndices_.resize(numGroups, std::vector<IndexType>(vocabSize, 0)); // [g][v] index of factor (or any valid index if it does not have it; we use 0)
  for (WordIndex v = 0; v < vocabSize; v++) {
    for (auto u : factorMap_[v]) {
      auto g = factorGroups_[u]; // convert u to relative u within factor group range
      ABORT_IF(u < groupRanges_[g].first || u >= groupRanges_[g].second, "Invalid factorGroups_ entry??");
      factorIndices_[g][v] = (IndexType)(u - groupRanges_[g].first);
      factorMasks_[g][v] = 1.0f;
    }
  }
  //for (Word v = 0; v < vocabSize; v++) {
  //  LOG(info, "'{}': {}*{} {}*{} {}*{} {}*{}", vocab[v],
  //      factorMasks_[0][v], factorIndices_[0][v],
  //      factorMasks_[1][v], factorIndices_[1][v],
  //      factorMasks_[2][v], factorIndices_[2][v],
  //      factorMasks_[3][v], factorIndices_[3][v]);
  //}
  for (size_t g = 0; g < numGroups; g++) { // detect non-overlapping groups
    LOG(info, "[embedding] Factor group '{}' has {} members ({})",
        groupPrefixes[g], groupCounts[g], groupCounts[g] == 1 ? "sigmoid" : "softmax");
    if (groupCounts[g] == 0) // factor group is unused  --@TODO: once this is not hard-coded, this is an error condition
      continue;
    ABORT_IF(groupRanges_[g].second - groupRanges_[g].first != groupCounts[g],
             "Factor group '{}' members should be consecutive in the factor vocabulary", groupPrefixes[g]);
  }

  // create the global factor matrix, which is used for factored embeddings
  std::vector<IndexType> data(vocabSize);
  std::iota(data.begin(), data.end(), 0);
  globalFactorMatrix_ = csr_rows(data); // [V x U]

  eosId_ = vocab_.getEosId();
  unkId_ = vocab_.getUnkId();

  ABORT_IF(maxSizeUnused != 0 && maxSizeUnused != size(), "Factored vocabulary does not allow on-the-fly clipping to a maximum vocab size (to {})", maxSizeUnused);
  return size();
}

/*virtual*/ Word FactoredVocab::operator[](const std::string& word) const /*override final*/ {
  ABORT("operator[] called indeed");
  return vocab_[word];
}

/*virtual*/ Words FactoredVocab::encode(const std::string& line, bool addEOS /*= true*/, bool inference /*= false*/) const /*override final*/ {
  return vocab_.encode(line, addEOS, inference);
}

/*virtual*/ std::string FactoredVocab::decode(const Words& sentence, bool ignoreEos /*= true*/) const /*override final*/ {
  return vocab_.decode(sentence, ignoreEos);
}

/*virtual*/ const std::string& FactoredVocab::operator[](Word id) const /*override final*/ {
  return vocab_[id];
}

/*virtual*/ size_t FactoredVocab::size() const /*override final*/ {
  return vocab_.size();
}

/*virtual*/ void FactoredVocab::createFake() /*override final*/ {
  return vocab_.createFake();
}

// create a CSR matrix M[V,U] from indices[] with
// M[v,u] = 1/c(u) if factor u is a factor of word v, and c(u) is how often u is referenced
FactoredVocab::CSRData FactoredVocab::csr_rows(const std::vector<IndexType>& words) const {
  std::vector<float> weights;
  std::vector<IndexType> indices;
  std::vector<IndexType> offsets;
  offsets.reserve(words.size() + 1);
  indices.reserve(words.size()); // (at least this many)
  // loop over all input words, and select the corresponding set of unit indices into CSR format
  offsets.push_back((IndexType)indices.size());
  for (auto v : words) {
    const auto& m = factorMap_[v];
    for (auto u : m) {
      indices.push_back(u);
      weights.push_back(1.0f/*/(float)factorRefCounts_[u]*/);
    }
    offsets.push_back((IndexType)indices.size()); // next matrix row begins at this offset
  }
  return { Shape({(int)words.size(), (int)factorVocab_.size()}), weights, indices, offsets };
}

Ptr<IVocab> createFactoredVocab(const std::string& vocabPath, Ptr<Options> options) {
  bool isFactoredVocab = regex::regex_search(vocabPath, regex::regex("\\.(fm)$"));
  if(isFactoredVocab)
    return New<FactoredVocab>(options);
  else
    return nullptr;
}

}  // namespace marian
