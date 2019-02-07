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

  // load and parse factorMap
  auto elements = factorShape_.elements();
  vocab_.resize(elements);
  factorMap_.resize(elements);
  auto factorVocabSize = factorVocab_.size();
  factorRefCounts_.resize(factorVocabSize);
  std::vector<std::string> tokens;
  std::string line;
  size_t numTotalFactors = 0;
  io::InputFileStream in(mapPath);
  for (WordIndex v = 0; io::getline(in, line); v++) {
    // parse the line, of the form WORD FACTOR1 FACTOR2 FACTOR1 ...
    // where FACTOR1 is the lemma, a factor that all words have.
    // Not every word has all other factors, so the n-th item is not always the same factor.
    utils::splitAny(line, tokens, " \t");
    ABORT_IF(tokens.size() < 2, "Factor map must have at least one factor per word", mapPath);
    std::vector<WordIndex> factorUnits;
    for (size_t i = 1/*first factor*/; i < tokens.size(); i++) {
      auto u = factorVocab_[tokens[i]];
      factorUnits.push_back(u);
      factorRefCounts_[u]++;
    }
    auto index = factorUnits2word(factorUnits).toWordIndex();
    // @TODO: map factors to non-dense integer
    factorMap_[index] = std::move(factorUnits);
    // add to vocab
    vocab_.add(tokens.front(), index);
    numTotalFactors += tokens.size() - 1;
    if (v % 5000 == 0)
      LOG(info, "{} -> {}", tokens.front(), word2string(Word::fromWordIndex(index)));
  }
  LOG(info, "[embedding] Factored-embedding map read with total/unique of {}/{} factors for {} valid words (in space of {})",
      numTotalFactors, factorVocabSize, vocab_.numValid(), size());

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
  ABORT_IF(maxSizeUnused != 0 && maxSizeUnused != size(), "Factored vocabulary does not allow on-the-fly clipping to a maximum vocab size (from {} to {})", size(), maxSizeUnused);
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
Word FactoredVocab::factors2word(const std::vector<size_t>& factorIndices /* [numGroups] */) {
  size_t index = 0;
  size_t numGroups = getNumGroups();
  ABORT_IF(factorIndices.size() != numGroups, "Factor indices array size must be same as number of factor groups");
  for (size_t g = 0; g < numGroups; g++) {
    auto factorIndex = factorIndices[g];
    if (factorIndex == FACTOR_NOT_APPLICABLE || factorIndex == FACTOR_NOT_SPECIFIED) // @TODO: check validity. If word has the factor, then N/A is invalid
      factorIndex = (size_t)factorShape_[g] - 1; // sentinel for "unused" or "not specified"
    else
      ABORT_IF(factorIndex >= (size_t)factorShape_[g] - 1, "Factor index out of range");
    index += factorIndex * factorStrides_[g];
  }
  return Word::fromWordIndex(index);
}

// like factors2word, except that factors are expressed as global unit indices, and result is just the WordIndex
// This is only used during initialization, so it's OK if it is a little inefficient.
Word FactoredVocab::factorUnits2word(const std::vector<WordIndex>& factorUnits) {
  // convert to fully unrolled factors representation
  std::vector<size_t> factorIndices(getNumGroups(), FACTOR_NOT_APPLICABLE); // default for unused factors
  for (auto u : factorUnits) {
    auto g = factorGroups_[u]; // convert u to relative u within factor group range
    ABORT_IF(u < groupRanges_[g].first || u >= groupRanges_[g].second, "Invalid factorGroups_ entry??");
    auto factorIndex = u - groupRanges_[g].first;
    factorIndices[g] = factorIndex;
  }
  return factors2word(factorIndices);
}

void FactoredVocab::word2factors(Word word, std::vector<size_t>& factorIndices /* [numGroups] */) {
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

std::string FactoredVocab::word2string(Word word) {
  std::vector<size_t> factorIndices;
  word2factors(word, factorIndices);
  std::string res;
  size_t numGroups = getNumGroups();
  for (size_t g = 0; g < numGroups; g++) {
    res.append(res.empty() ? "(" : ", ");
    auto factorIndex = factorIndices[g];
    switch (factorIndex) {
    case FACTOR_NOT_APPLICABLE: res.append("n/a"); break;
    case FACTOR_NOT_SPECIFIED: res.append("?"); break;
    default: res.append(factorVocab_[(WordIndex)(factorIndex + groupRanges_[g].first)]); break;
    }
  }
  return res + ")";
}

size_t FactoredVocab::getFactor(Word word, size_t groupIndex) {
  size_t index = word.toWordIndex();
  index = index / factorStrides_[groupIndex];
  index = index % (size_t)factorShape_[groupIndex];
  if (index == (size_t)factorShape_[groupIndex] - 1) {
    index = FACTOR_NOT_APPLICABLE; // @BUGBUG: We should check here which one it is.
  }
  return index;
}

std::pair<WordIndex, bool> FactoredVocab::getFactorUnit(Word word, size_t groupIndex) {
  word; groupIndex;
  ABORT("Not implemented");
}

void FactoredVocab::constructNormalizationInfoForVocab() {
  // create mappings needed for normalization in factored outputs
  size_t numGroups = groupPrefixes_.size();
  size_t vocabSize = vocab_.size();
  factorMasks_  .resize(numGroups, std::vector<float>(vocabSize, 0));     // [g][v] 1.0 if word v has factor g
  factorIndices_.resize(numGroups, std::vector<IndexType>(vocabSize, 0)); // [g][v] index of factor (or any valid index if it does not have it; we use 0)
  gapLogMask_.resize(vocabSize, -1e8f);
  for (WordIndex v = 0; v < vocabSize; v++) {
    for (auto u : factorMap_[v]) {
      auto g = factorGroups_[u]; // convert u to relative u within factor group range
      ABORT_IF(u < groupRanges_[g].first || u >= groupRanges_[g].second, "Invalid factorGroups_ entry??");
      factorIndices_[g][v] = (IndexType)(u - groupRanges_[g].first);
      factorMasks_[g][v] = 1.0f;
      gapLogMask_[v] = 0.0f; // valid entry
    }
  }
  //for (Word v = 0; v < vocabSize; v++) {
  //  LOG(info, "'{}': {}*{} {}*{} {}*{} {}*{}", vocab[v],
  //      factorMasks_[0][v], factorIndices_[0][v],
  //      factorMasks_[1][v], factorIndices_[1][v],
  //      factorMasks_[2][v], factorIndices_[2][v],
  //      factorMasks_[3][v], factorIndices_[3][v]);
  //}

  // create the global factor matrix, which is used for getLogits() only
  std::vector<IndexType> data(vocabSize);
  std::iota(data.begin(), data.end(), 0);
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

/*virtual*/ const std::string& FactoredVocab::operator[](Word id) const /*override final*/ {
  return vocab_[id.toWordIndex()];
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
  for (auto w : words) {
    const auto& m = factorMap_[w];
    for (auto u : m) {
      indices.push_back(u);
      weights.push_back(1.0f/*/(float)factorRefCounts_[u]*/);
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
  ABORT_IF(word.empty(), "Invalid access to dictionary gap item");
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
