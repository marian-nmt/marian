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

  // load factor vocabulary
  factorVocab_.load(factorVocabPath);
  groupPrefixes_ = { "(lemma)", "@C", "@GL", "@GR" }; // @TODO: hard-coded for these initial experiments

  // construct mapping tables for factors
  constructGroupInfoFromFactorVocab();
  constructFactorIndexConversion();
  auto numGroups = getNumGroups();

  // load and parse factorMap
  auto vocabSize = factorShape_.elements(); // size of vocab space including gaps
  vocab_.resize(vocabSize);
  //factorMap_.resize(vocabSize);
  auto factorVocabSize = factorVocab_.size();
  lemmaHasFactorGroup_.resize(groupRanges_[0].second - groupRanges_[0].first);
  std::vector<std::string> tokens;
  std::string line;
  size_t numTotalFactors = 0;
  io::InputFileStream in(mapPath);
  for (WordIndex v = 0; io::getline(in, line); v++) {
    // parse the line, of the form WORD FACTOR1 FACTOR2 FACTOR1 ...
    // where FACTOR1 is the lemma, a factor that all words have.
    // Not every word has all other factors, so the n-th item is not always in the same factor group.
    // @TODO: change to just use the .wl file, and manually split at @
    utils::splitAny(line, tokens, " \t");
    ABORT_IF(tokens.size() < 2, "Factor map must have at least one factor per word", mapPath);
    std::vector<WordIndex> factorUnits;
    for (size_t i = 1/*first factor*/; i < tokens.size(); i++) {
      auto u = factorVocab_[tokens[i]];
      factorUnits.push_back(u);
    }
    // convert to fully unrolled factors representation
    std::vector<size_t> factorIndices(groupRanges_.size(), FACTOR_NOT_APPLICABLE); // default for unused factors
    std::vector<bool> hasFactorGroupFlags(groupRanges_.size(), false);
    for (auto u : factorUnits) {
      factorIndices[factorGroups_[u]] = factorUnit2FactorIndex(u);
      hasFactorGroupFlags[factorGroups_[u]] = true;
    }
    // record which lemma has what factor groups
    ABORT_IF(!hasFactorGroupFlags[0], "Factor map does not specify a lemma (factor of first group) for word {}", tokens.front());
    auto& lemmaFlags = lemmaHasFactorGroup_[factorIndices[0]];
    if (lemmaFlags.empty())
      lemmaFlags = std::move(hasFactorGroupFlags);
    else
      ABORT_IF(lemmaFlags != hasFactorGroupFlags, "Inconsistent factor groups used for word {}", tokens.front());
    // map factors to non-dense integer
    auto word = factors2word(factorIndices);
    auto wordIndex = word.toWordIndex();
    //factorMap_[wordIndex] = std::move(factorUnits);
    // add to vocab (the wordIndex are not dense, so the vocab will have holes)
    vocab_.add(tokens.front(), wordIndex);
    numTotalFactors += tokens.size() - 1;
    if (v % 5000 == 0)
      LOG(info, "{} -> {}", tokens.front(), word2string(word));
  }
  LOG(info, "[embedding] Factored-embedding map read with total/unique of {}/{} factors for {} valid words (in space of {})",
      numTotalFactors, factorVocabSize, vocab_.numValid(), size());

  // enumerate all combinations of factors for each lemma
  // @TODO: switch to factor-spec, which no longer enumerates all combinations. Then don't set vocab string here.
  for (size_t v = 0; v < vocabSize; v++) {
    auto word = Word::fromWordIndex(v);
    bool isValid = true;
    for (size_t g = 0; isValid && g < numGroups; g++) {
      auto factorIndex = getFactor(word, g);
      // @TODO: we have a hack in getFactor() to return not-specified if factor is specified but not applicable, making it invalid
      isValid = factorIndex != FACTOR_NOT_SPECIFIED; // FACTOR_NOT_APPLICABLE is a valid value
    }
    if (isValid != !vocab_.isGap((WordIndex)v))
    {
      //LOG(info, "WARNING: Factored vocab mismatch for {}: isValid={}, isGap={}", word2string(word), isValid, vocab_.isGap((WordIndex)v));
      if (isValid) // add the missing word (albeit with a poor grapheme)
        (*this)[word];
    }
  }

  // create mappings needed for normalization in factored outputs
  constructNormalizationInfoForVocab();

  // </s> and <unk> must exist in the vocabulary
  eosId_ = Word::fromWordIndex(vocab_[DEFAULT_EOS_STR]);
  unkId_ = Word::fromWordIndex(vocab_[DEFAULT_UNK_STR]);
  LOG(info, "eos: {}; unk: {}", word2string(eosId_), word2string(unkId_));

#if 1   // dim-vocabs stores numValid() in legacy model files, and would now have been size()
  if (maxSizeUnused == vocab_.numValid())
    maxSizeUnused = vocab_.size();
#endif
  //ABORT_IF(maxSizeUnused != 0 && maxSizeUnused != size(), "Factored vocabulary does not allow on-the-fly clipping to a maximum vocab size (from {} to {})", size(), maxSizeUnused);
  // @TODO: ^^ disabled now that we are generating the full combination of factors; reenable once we have consistent setups again
  return size();
}

void FactoredVocab::constructGroupInfoFromFactorVocab() {
  // form groups
  size_t numGroups = groupPrefixes_.size();
  size_t factorVocabSize = factorVocab_.size();
  factorGroups_.resize(factorVocabSize, 0);
  for (size_t g = 1; g < groupPrefixes_.size(); g++) { // set group labels; what does not match any prefix will stay in group 0
    const auto& groupPrefix = groupPrefixes_[g];
    for (WordIndex u = 0; u < factorVocabSize; u++)
      if (utils::beginsWith(factorVocab_[u], groupPrefix)) {
        ABORT_IF(factorGroups_[u] != 0, "Factor {} matches multiple groups, incl. {}", factorVocab_[u], groupPrefix);
        factorGroups_[u] = g;
      }
  }
  // determine group index ranges
  groupRanges_.resize(numGroups, { SIZE_MAX, (size_t)0 });
  std::vector<int> groupCounts(numGroups); // number of group members
  for (WordIndex u = 0; u < factorVocabSize; u++) { // determine ranges; these must be non-overlapping, verified via groupCounts
    auto g = factorGroups_[u];
    if (groupRanges_[g].first > u)
        groupRanges_[g].first = u;
    if (groupRanges_[g].second < u + 1)
        groupRanges_[g].second = u + 1;
    groupCounts[g]++;
  }
  for (size_t g = 0; g < numGroups; g++) { // detect non-overlapping groups
    LOG(info, "[embedding] Factor group '{}' has {} members", groupPrefixes_[g], groupCounts[g]);
    if (groupCounts[g] == 0) // factor group is unused  --@TODO: once this is not hard-coded, this is an error condition
      continue;
    ABORT_IF(groupRanges_[g].second - groupRanges_[g].first != groupCounts[g],
             "Factor group '{}' members should be consecutive in the factor vocabulary", groupPrefixes_[g]);
  }
  // we map between factors and flat WordIndex like indexing a tensor
  constructFactorIndexConversion();
}

// create factorShape_ and factorStrides_, for mapping between flat (non-dense) ids and factor arrays
void FactoredVocab::constructFactorIndexConversion() {
  std::vector<int> shape;
  for (const auto& r : groupRanges_)
    shape.push_back((int)(r.second - r.first + 1)); // +1 to reserve the last value for either "factor not used" or "factor not present"
  factorShape_ = Shape(std::move(shape));
  factorStrides_.resize(factorShape_.size(), 1);
  for (size_t g = factorStrides_.size() - 1; g --> 0; )
    factorStrides_[g] = factorStrides_[g + 1] * (size_t)factorShape_[g + 1];
}

// encode factors into a Word struct
Word FactoredVocab::factors2word(const std::vector<size_t>& factorIndices /* [numGroups] */) const {
  size_t index = 0;
  size_t numGroups = getNumGroups();
  ABORT_IF(factorIndices.size() != numGroups, "Factor indices array size must be same as number of factor groups");
  for (size_t g = 0; g < numGroups; g++) {
    auto factorIndex = factorIndices[g];
    if (factorIndex != FACTOR_NOT_SPECIFIED) { // check validity
      auto factor0Index = factorIndices[0]; // lemma
      ABORT_IF(factor0Index == FACTOR_NOT_SPECIFIED, "Without lemma, no other factor may be specified");
      ABORT_IF(lemmaHasFactorGroup(factor0Index, g) == (factorIndex == FACTOR_NOT_APPLICABLE), "Lemma {} does not have factor group {}", factor0Index, g);
    }
    if (factorIndex == FACTOR_NOT_APPLICABLE || factorIndex == FACTOR_NOT_SPECIFIED)
      factorIndex = (size_t)factorShape_[g] - 1; // sentinel for "unused" or "not specified"
    else
      ABORT_IF(factorIndex >= (size_t)factorShape_[g] - 1, "Factor index out of range");
    index += factorIndex * factorStrides_[g];
  }
  return Word::fromWordIndex(index);
}

Word FactoredVocab::lemma2Word(size_t factor0Index) const {
  size_t numGroups = getNumGroups();
  std::vector<size_t> factorIndices;
  factorIndices.reserve(numGroups);
  factorIndices.push_back(factor0Index);
  for (size_t g = 1; g < numGroups; g++) {
    auto index = lemmaHasFactorGroup(factor0Index, g) ? FACTOR_NOT_SPECIFIED : FACTOR_NOT_APPLICABLE;
    factorIndices.push_back(index);
  }
  return factors2word(factorIndices);
}

// replace a factor that is FACTOR_NOT_SPECIFIED by a specified one
// This is used in beam search, where factors are searched one after another.
Word FactoredVocab::expandFactoredWord(Word word, size_t groupIndex, size_t factorIndex) const {
  //LOG(info, "expand {} + [{}]={}", word2string(word), groupIndex, factorIndex);
  ABORT_IF(groupIndex == 0, "Cannot add or change lemma in a partial Word");
  ABORT_IF(!isFactorValid(factorIndex), "Cannot add unspecified or n/a factor to a partial Word");
  std::vector<size_t> factorIndices;
  word2factors(word, factorIndices);
  auto factor0Index = factorIndices[0];
  ABORT_IF(!isFactorValid(factor0Index), "Cannot add factor to a partial Word without lemma");
  ABORT_IF(factorIndices[groupIndex] == FACTOR_NOT_APPLICABLE, "Cannot add a factor that the lemma does not have");
  ABORT_IF(factorIndices[groupIndex] != FACTOR_NOT_SPECIFIED, "Cannot modify a specified factor in a partial Word");
  factorIndices[groupIndex] = factorIndex;
  word = factors2word(factorIndices);
  //LOG(info, "to {}", word2string(word));
  return word;
}

size_t FactoredVocab::factorUnit2FactorIndex(WordIndex u) const {
  auto g = factorGroups_[u]; // convert u to relative u within factor group range
  ABORT_IF(u < groupRanges_[g].first || u >= groupRanges_[g].second, "Invalid factorGroups_ entry??");
  return u - groupRanges_[g].first;
}


void FactoredVocab::word2factors(Word word, std::vector<size_t>& factorIndices /* [numGroups] */) const {
  size_t numGroups = getNumGroups();
  factorIndices.resize(numGroups);
  for (size_t g = 0; g < numGroups; g++) {
    auto factorIndex = getFactor(word, g);
    factorIndices[g] = factorIndex;
  }
#if 1
  auto test = factors2word(factorIndices);
  ABORT_IF(test != word, "Word <-> factor conversion broken??");
#endif
}

std::string FactoredVocab::word2string(Word word) const {
  // this function has some code dup, so that we can bypass some checks for debugging
  size_t numGroups = getNumGroups();
  size_t factor0Index = word.toWordIndex() / factorStrides_[0];
  std::string res;
  for (size_t g = 0; g < numGroups; g++) {
    res.append(res.empty() ? "(" : ", ");
    size_t index = word.toWordIndex();
    index = index / factorStrides_[g];
    index = index % (size_t)factorShape_[g];
    if (index == (size_t)factorShape_[g] - 1) { // special sentinel value for unspecified or not-applicable
      if (factor0Index >= (size_t)factorShape_[0])
        res.append("(lemma oob)");
      else if (lemmaHasFactorGroup(factor0Index, g))
        res.append("?");
      else
        res.append("n/a");
    }
    else
      res.append(factorVocab_[(WordIndex)(index + groupRanges_[g].first)]);
  }
  return res + ")";
}

size_t FactoredVocab::getFactor(Word word, size_t groupIndex) const {
  size_t index = word.toWordIndex();
  size_t factor0Index = index / factorStrides_[0];
  index = index / factorStrides_[groupIndex];
  index = index % (size_t)factorShape_[groupIndex];
  if (index == (size_t)factorShape_[groupIndex] - 1) { // special sentinel value for unspecified or not-applicable
    if (groupIndex == 0) // lemma itself is always applicable, hence 'not specified'
      index = FACTOR_NOT_SPECIFIED;
    else { // not lemma: check whether lemma of word has this factor group
      if (lemmaHasFactorGroup(factor0Index, groupIndex))
        index = FACTOR_NOT_SPECIFIED;
      else
        index = FACTOR_NOT_APPLICABLE;
    }
  }
  else { // regular value: consistency check if lemma really has this factor group
    ABORT_IF(factor0Index == (size_t)factorShape_[0] - 1, "Word has specified factor but no lemma??");
    //ABORT_IF(!lemmaHasFactorGroup(factor0Index, groupIndex), "Word has a specified factor for a lemma that does not have that factor group??");
    if (!lemmaHasFactorGroup(factor0Index, groupIndex))
      index = FACTOR_NOT_SPECIFIED;
    // @TODO: ^^ needed for determining all valid vocab entries; can we pass a flag in to allow this?
  }
  return index;
}

//std::pair<WordIndex, bool> FactoredVocab::getFactorUnit(Word word, size_t groupIndex) const {
//  word; groupIndex;
//  ABORT("Not implemented");
//}

void FactoredVocab::constructNormalizationInfoForVocab() {
  // create mappings needed for normalization in factored outputs
  //size_t numGroups = groupPrefixes_.size();
  size_t vocabSize = size();
  //factorMasks_  .resize(numGroups, std::vector<float>(vocabSize, 0));     // [g][v] 1.0 if word v has factor g
  //factorIndices_.resize(numGroups, std::vector<IndexType>(vocabSize, 0)); // [g][v] index of factor (or any valid index if it does not have it; we use 0)
  gapLogMask_.resize(vocabSize, -1e8f);
  for (WordIndex v = 0; v < vocabSize; v++) {
#if 1 // @TODO: TEST THIS again by disabling factored decoding in beam_search.h
    if (!vocab_.isGap(v))
      gapLogMask_[v] = 0.0f; // valid entry
#else
    for (auto u : factorMap_[v]) {
      auto g = factorGroups_[u]; // convert u to relative u within factor group range
      ABORT_IF(u < groupRanges_[g].first || u >= groupRanges_[g].second, "Invalid factorGroups_ entry??");
      //factorIndices_[g][v] = (IndexType)(u - groupRanges_[g].first);
      //factorMasks_[g][v] = 1.0f;
      gapLogMask_[v] = 0.0f; // valid entry
    }
#endif
  }
  //for (Word v = 0; v < vocabSize; v++) {
  //  LOG(info, "'{}': {}*{} {}*{} {}*{} {}*{}", vocab[v],
  //      factorMasks_[0][v], factorIndices_[0][v],
  //      factorMasks_[1][v], factorIndices_[1][v],
  //      factorMasks_[2][v], factorIndices_[2][v],
  //      factorMasks_[3][v], factorIndices_[3][v]);
  //}

  // create the global factor matrix, which is used for getLogits() only
  // For invalid words, this leaves empty matrix rows, which are later masked by adding gapLogMask.
  Words data;
  for (size_t v = 0; v < vocabSize; v++) // note: this loops over the entire vocab space, incl. gaps
    data.push_back(Word::fromWordIndex(v));
  globalFactorMatrix_ = csr_rows(data); // [V x U]
}

/*virtual*/ Word FactoredVocab::operator[](const std::string& word) const /*override final*/ {
  WordIndex index;
  bool found = vocab_.tryFind(word, index);
  if (found)
    return Word::fromWordIndex(index);
  else
    return getUnkId();
}

/*virtual*/ const std::string& FactoredVocab::operator[](Word word) const /*overrworde final*/ {
  //LOG(info, "Looking up Word {}={}", word.toWordIndex(), word2string(word));
#if 1 // @BUGBUG: our manually prepared dict does not contain @CI tags for single letters, but it's a valid factor
  if (vocab_.isGap(word.toWordIndex())) {
    LOG_ONCE(info, "Factor combination {} missing in external dict, generating fake entry (only showing this warning once)", word2string(word));
    const_cast<WordLUT&>(vocab_).add("??" + word2string(word), word.toWordIndex());
  }
#endif
  return vocab_[word.toWordIndex()];
}

/*virtual*/ size_t FactoredVocab::size() const /*override final*/ {
  return vocab_.size();
}

/*virtual*/ Words FactoredVocab::encode(const std::string& line, bool addEOS /*= true*/, bool /*inference*/ /*= false*/) const /*override final*/ {
  std::vector<std::string> lineTokens;
  utils::split(line, lineTokens, " ");
  Words res; res.reserve(lineTokens.size() + addEOS);
  for (const auto& tok : lineTokens)
    res.push_back((*this)[tok]);
  if (addEOS)
    res.push_back(getEosId());
  return res;
}

/*virtual*/ std::string FactoredVocab::decode(const Words& sentence, bool ignoreEOS /*= true*/) const /*override final*/ {
  std::vector<std::string> decoded;
  decoded.reserve(sentence.size());
  for(auto w : sentence) {
    if((w != getEosId() || !ignoreEOS))
      decoded.push_back((*this)[w]);
  }
  return utils::join(decoded, " ");
}

// create a CSR matrix M[V,U] from words[] with
// M[v,u] = 1/c(u) if factor u is a factor of word v, and c(u) is how often u is referenced
FactoredVocab::CSRData FactoredVocab::csr_rows(const Words& words) const {
  auto numGroups = getNumGroups();
  std::vector<float> weights;
  std::vector<IndexType> indices;
  std::vector<IndexType> offsets;
  offsets.reserve(words.size() + 1);
  indices.reserve(words.size()); // (at least this many)
  // loop over all input words, and select the corresponding set of unit indices into CSR format
  offsets.push_back((IndexType)indices.size());
  std::vector<size_t> factorIndices;
  for (auto word : words) {
    if (!vocab_.isGap(word.toWordIndex())) { // skip invalid combinations in the space (can only happen during initialization)  --@TODO: add a check?
      word2factors(word, factorIndices);
#if 0 // original code; enable this to try
      numGroups;
      const auto& m = factorMap_[word.toWordIndex()];
      for (auto u : m) {
        indices.push_back(u);
#else
#if 1 // special handling of the missing single capitalized letters
      // costs about 0.1 BLEU when original model never saw this combination (which is quite nicely low)
      // @TODO: remove this once we use the factor-spec file
      const auto& lemma = factorVocab_[(WordIndex)(factorIndices[0] + groupRanges_[0].first)];
      if (lemma.size() == 1 && factorIndices[1]/*@C*/ == 1/*@CI*/) // skip one-letter factors with 
          LOG_ONCE(info, "Suppressing embedding for word {} (only showing this warning once)", word2string(word));
      else
#endif
      for (size_t g = 0; g < numGroups; g++) { // @TODO: make this faster by having a list of all factors to consider for a lemma?
        auto factorIndex = factorIndices[g];
        ABORT_IF(factorIndex == FACTOR_NOT_SPECIFIED, "Attempted to embed a word with a factor not specified");
        if (factorIndex == FACTOR_NOT_APPLICABLE)
          continue;
        indices.push_back((IndexType)(factorIndex + groupRanges_[g].first)); // map to unit index
#endif
        weights.push_back(1.0f);
      }
    }
    offsets.push_back((IndexType)indices.size()); // next matrix row begins at this offset
  }
  return { Shape({(int)words.size(), (int)factorVocab_.size()}), weights, indices, offsets };
}

// Helper to construct and load a FactordVocab from a path is given (non-empty) and if it specifies a factored vocab.
// This is used by the Embedding and Output layers.
/*static*/ Ptr<FactoredVocab> FactoredVocab::tryCreateAndLoad(const std::string& path) {
  Ptr<FactoredVocab> res;
  if (!path.empty()) {
    res = std::static_pointer_cast<FactoredVocab>(createFactoredVocab(path)); // this checks the file extension
    if (res)
      res->load(path); // or throw
  }
  return res;
}

// WordLUT
WordIndex FactoredVocab::WordLUT::add(const std::string& word, WordIndex index) {
  ABORT_IF(word.empty(), "Attempted to add the empty word to a dictionary");
  auto wasInserted = str2index_.insert(std::make_pair(word, index)).second;
  ABORT_IF(!wasInserted, "Duplicate vocab entry for '{}'", word);
  while (index2str_.size() <= index)
    index2str_.emplace_back(); // @TODO: what's the right way to get linear complexity in steps?
  ABORT_IF(!index2str_[index].empty(), "Duplicate vocab entry for index {} (new: '{}'; existing: '{}')", index, word, index2str_[index]);
  index2str_[index] = word;
  return index;
}
const std::string& FactoredVocab::WordLUT::operator[](WordIndex index) const {
  const auto& word = index2str_[index];
  //ABORT_IF(word.empty(), "Invalid access to dictionary gap item");
  return word;
}
WordIndex FactoredVocab::WordLUT::operator[](const std::string& word) const {
  auto iter = str2index_.find(word);
  ABORT_IF(iter == str2index_.end(), "Token '{}' not found in vocabulary", word);
  return iter->second;
}
bool FactoredVocab::WordLUT::tryFind(const std::string& word, WordIndex& index) const {
  auto iter = str2index_.find(word);
  if (iter == str2index_.end())
    return false;
  index = iter->second;
  return true;
}
void FactoredVocab::WordLUT::resize(size_t num) {
  ABORT_IF(num < index2str_.size(), "Word table cannot be shrunk");
  index2str_.resize(num); // gets filled up with gap items (empty strings)
}
size_t FactoredVocab::WordLUT::load(const std::string& path) {
  std::string line;
  io::InputFileStream in(path);
  for (WordIndex v = 0; io::getline(in, line); v++)
    add(line, v);
  return size();
}

// Note: This does not actually load it, only checks the path for the type.
Ptr<IVocab> createFactoredVocab(const std::string& vocabPath) {
  bool isFactoredVocab = regex::regex_search(vocabPath, regex::regex("\\.(fm)$"));
  if(isFactoredVocab)
    return New<FactoredVocab>();
  else
    return nullptr;
}

}  // namespace marian
