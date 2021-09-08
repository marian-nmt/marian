// This is the main implementation of factored models, which are driven by the vocabulary.
// Decoding, embedding, and output layer call into the vocab to drive their behavior.

#include "data/vocab_base.h"
#include "common/definitions.h"
#include "data/types.h"
#include "common/regex.h"
#include "data/factored_vocab.h"
#include <set>

// @TODO: review all comments and clarify nomenclature:
// * factor type (e.g. caps: |c* ); currently called a "group"
// * factor name (e.g. all-caps: |ca )
// * factor index (e.g. |ca is index 0 inside |ca |ci |cn)
// * factor unit index (|ca is unit 41324 in joint factor vocab)
// Also remove references to older outdated versions.

namespace marian {

/*virtual*/ size_t FactoredVocab::load(const std::string& modelPath, size_t maxSizeUnused /*= 0*/) /*override final*/ {
  maxSizeUnused;
  // If model has already been loaded, then assume this is a shared object, and skip loading it again.
  // This can be multi-threaded, so must run under lock.
  static std::mutex s_mtx;
  std::lock_guard<std::mutex> criticalSection(s_mtx);
  if (size() != 0) {
    //LOG(info, "[vocab] Attempting to load model a second time; skipping (assuming shared vocab)");
    return size();
  }
  LOG(info, "[vocab] Loading vocab spec file {}", modelPath);

  // load factor-vocab file and parse it
  std::vector<std::vector<std::string>> factorMapTokenized;
  std::string line;
  std::vector<std::string> tokBuf;
  if (utils::endsWith(modelPath, ".fsv")) { // @TODO: this extension check is only for backcompat; can be removed once we no longer support the old format
    // this is a fake parser for the generic factor spec, which makes a few hard assumptions:
    //     - all types starting with _ except _has_* are factor names
    //     - X : _x makes surface form X part of prob distribution _x except for _has_*
    //     - X : _has_x adds factor "x" to lemma X
    //     - _x <-> form only allows "_x <->" or "_x <-> _has_x" (same x), and is otherwise unused
    //     - _lemma is special
    // The current version of the code just converts it internally to the legacy form.
    // @TODO: Once the legacy form is no longer needed, simplify this.
    io::InputFileStream in(modelPath);
    WordIndex v = 0;
    std::map<std::string,std::set<std::string>> factorTypeMap; // [type name] -> {factor-type names}
    std::vector<std::string> deferredFactorVocab; // factor surface forms are presently expected to be at the end of factorVocab_, so collect them here first
    while(io::getline(in, line)) {
#if 1 // workaround for a bug fix in FactoredSegmenter that made old .fsv files incompatible
      if (line      == "\xef\xb8\x8f : _lemma _has_wb")         // old vocabs have a wrong factor in here
        line         = "\xef\xb8\x8f : _lemma _has_gl _has_gr"; // patch it to the correct one
      else if (line == "\xef\xb8\x8e : _lemma _has_wb")
        line         = "\xef\xb8\x8e : _lemma _has_gl _has_gr";
#endif
      utils::splitAny(line, tokBuf, " \t");
      if (tokBuf.empty() || tokBuf[0][0] == '#') // skip comments and blank lines
        continue;
      const auto& lhs = tokBuf[0];
      const auto& op = tokBuf.size() > 1 ? tokBuf[1] : "";
      if (lhs[0] == '_') { // factor name
        if (utils::beginsWith(lhs, "_has_")) {
          const auto fName = lhs.substr(5); // skip _has_
          ABORT_IF(factorTypeMap.find(fName) == factorTypeMap.end(), "Factor trait '{}' requires a factor named '{}' to exist", lhs, fName);
          ABORT_IF(tokBuf.size() != 1, "Extraneous characters after factor trait: '{}'", line);
          continue;
        }
        else if (op == "<->") {
          ABORT_IF(lhs == "_lemma" && tokBuf.size() != 2, "Lemma factor distribution cannot be conditioned: '{}'", line);
          ABORT_IF(lhs != "_lemma" && (tokBuf.size() != 3 || tokBuf[2] != "_has" + lhs), "Factor distribution can only be conditioned on nothing or on _has{}: '{}'", lhs, line);
          continue;
        }
        else { // this declares a new factor
          ABORT_IF(tokBuf.size() != 1, "Extraneous characters after factor declaration: '{}'", line);
          const auto& fName = lhs.substr(1); // skip _
          ABORT_IF(factorTypeMap.empty() && fName != "lemma", "First factor must be _lemma");
          auto rv = factorTypeMap.insert(std::make_pair(fName, std::set<std::string>())); // create new factor
          ABORT_IF(!rv.second, "Factor declared twice: '{}'", line);
          groupPrefixes_.push_back(fName == "lemma" ? "(lemma)" : ("|" + fName));
          continue;
        }
      }
      else { // if not _ then it is a surface form
        ABORT_IF(op != ":" || 2 >= tokBuf.size(), "Factor-lemma declaration should have the form LEMMA : _FACTOR, _has_FACTOR, _has_FACTOR... in '{}'", line);
        ABORT_IF(tokBuf[2][0] != '_', "Factor name should begin with _ in '{}'", line);
        ABORT_IF(utils::beginsWith(tokBuf[2], "_has_"), "The first factor after : must not begin with _has_ in '{}'", line);
        // add to surface-form dictionary
        const auto& fName = tokBuf[2].substr(1); // skip _
        auto isLemma = fName == "lemma";
        if (isLemma)
          factorVocab_.add(lhs, v++); // note: each item can only be declared once
        else
          deferredFactorVocab.push_back(lhs);        // add surface form to its declared factor type
        auto surfaceFormSet = factorTypeMap.find(fName); // set of surface forms for this factor
        ABORT_IF(surfaceFormSet == factorTypeMap.end(), "Unknown factor name in '{}'", line);
        auto rv = surfaceFormSet->second.insert(lhs); // insert surface form into its declared factor type
        ABORT_IF(!rv.second, "Factor declared twice: '{}'", line);
        auto tokenizedMapLine = isLemma ? std::vector<std::string>{ lhs, lhs } : std::vector<std::string>();
        // associated factors
        for (size_t i = 3; i < tokBuf.size(); i++) {
          const auto& has = tokBuf[i];
          ABORT_IF(!utils::beginsWith(has, "_has_"), "Factor associations must use the form _has_X in '{}'", line);
          ABORT_IF(!isLemma, "Factor associations are only allowed when factor type is _lemma: '{}', line");
          const auto& faName = has.substr(5); // skip _has_ and prepend |
          // for tokenized map, we pick one example of the factor names
          auto iter = factorTypeMap.find(faName);
          ABORT_IF(iter == factorTypeMap.end(), "Invalid factor association {}, no such factor: '{}'", has, line);
          const auto& factorNames = iter->second;
          ABORT_IF(factorNames.empty(), "Factor association {} refers to empty factor type: '{}'", has, line);
          const auto& oneFactorName = "|" + *factorNames.begin(); // pick the first entry as one example
          tokenizedMapLine[0] += oneFactorName;
          tokenizedMapLine.push_back(oneFactorName);
        }
        if (isLemma)
          factorMapTokenized.push_back(std::move(tokenizedMapLine));
        continue;
      }
      ABORT("Malformed .fsv input line {}", line); // we only get here for lines we could not process
    }
    for (auto factorTypeName : deferredFactorVocab)
      factorVocab_.add("|" + factorTypeName, v++);
  } else {  // legacy for old configs
    // legacy format: one factor map, one flat list of factor surface forms
    // load factor vocabulary
    factorSeparator_ = '@';
    auto factorVocabPath = modelPath;
    factorVocabPath.back() = 'l'; // map .fm to .fl
    factorVocab_.load(factorVocabPath);
    groupPrefixes_ = { "(lemma)", "@C", "@GL", "@GR", "@WB"/*, "@WE"*/, "@CB"/*, "@CE"*/ }; // @TODO: hard-coded for these initial experiments
    // @TODO: add checks for empty factor groups until it stops crashing (training already works; decoder still crashes)

    io::InputFileStream in(modelPath);
    for (WordIndex v = 0; io::getline(in, line); v++) {
      utils::splitAny(line, tokBuf, " \t");
      factorMapTokenized.push_back(tokBuf);
    }
  }

  // construct mapping tables for factors
  constructGroupInfoFromFactorVocab();
  constructFactorIndexConversion();

  // parse factorMap
  // modelPath = path to file with entries in order of vocab entries of the form
  //   WORD FACTOR1 FACTOR2 FACTOR3...
  // Factors are grouped
  //  - user specifies list-factor prefixes; all factors beginning with that prefix are in the same group
  //  - factors within a group as multi-class and normalized that way
  //  - groups of size 1 are interpreted as sigmoids, multiply with P(u) / P(u-1)
  //  - one prefix must not contain another
  //  - all factors not matching a prefix get lumped into yet another class (the lemmas)
  //  - factor vocab must be sorted such that all groups are consecutive
  //  - result of Output layer is nevertheless logits, not a normalized probability, due to the sigmoid entries
  // For every lemma, the factor map contains one example. At the end of this loop, we have a vocabulary
  // vocab_ that contains those examples, but not all possible combinations
  lemmaHasFactorGroup_.resize(groupRanges_[0].second - groupRanges_[0].first); // group 0 is the lemmas; this difference is the number of lemma symbols
  size_t numTotalFactors = 0;
  for (WordIndex v = 0; v < factorMapTokenized.size(); v++) {
    const auto& tokens = factorMapTokenized[v];
    // parse the line, of the form WORD FACTOR1 FACTOR2 FACTOR1 ...
    // where FACTOR1 is the lemma, a factor that all words have.
    // Not every word has all other factors, so the n-th item is not always in the same factor group.
    // @TODO: change to just use the .wl file, and manually split at @
    ABORT_IF(tokens.size() < 2, "Factor map must have at least one factor per word", modelPath);
    std::vector<WordIndex> factorUnits; // units in the joint factor vocab that belong to a specific factor type
    for (size_t i = 1/*first factor*/; i < tokens.size(); i++) {
      auto u = factorVocab_[tokens[i]];
      factorUnits.push_back(u);
    }
    // convert to fully unrolled factors representation
    auto na = FACTOR_NOT_APPLICABLE; // (gcc compiler bug: sometimes it cannot find this if passed directly)
    std::vector<size_t> factorIndices(groupRanges_.size(), na); // default for unused factors
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
    // add to vocab (the wordIndex are not dense, so the vocab will have holes)
    // for now add what we get, and then expand more below
    auto wordString = word2string(word);
    if (tokens.front() != wordString) // order may differ, since we formed the input based on the factors in the user file, which may be in any order
      LOG_ONCE(info, "[vocab] Word name in vocab file {} differs from canonical form {} (this warning is only shown once)", tokens.front(), wordString);
    vocab_.add(wordString, word.toWordIndex());
    numTotalFactors += tokens.size() - 1;
  }
  LOG(info, "[vocab] Factored-embedding map read with total/unique of {}/{} factors from {} example words (in space of {})",
      numTotalFactors, factorVocabSize(), vocab_.size()/*numValid()*/, utils::withCommas(virtualVocabSize()));
  //vocab_.dumpToFile(modelPath + "_examples");

  // enumerate all valid combinations of factors for each lemma and add them to vocab_
  // Having vocab_ makes life easier, although it is not strictly needed. Typical expanded valid vocabs
  // are on the order of 200k entries. If we ever go much larger, we'd want to elimimate vocab_
  // and fully virtualize its function.
  LOG(info, "[vocab] Expanding all valid vocab entries out of {}...", utils::withCommas(virtualVocabSize()));
  std::vector<size_t> factorIndices(getNumGroups());
  rCompleteVocab(factorIndices, /*g=*/0);
  LOG(info, "[vocab] Completed, total {} valid combinations", vocab_.size()/*numValid()*/);
  //vocab_.dumpToFile(modelPath + "_expanded");

#ifdef FACTOR_FULL_EXPANSION
  // create mappings needed for normalization in factored outputs
  constructNormalizationInfoForVocab();
#endif

  // </s> and <unk> must exist in the vocabulary
  eosId_ = Word::fromWordIndex(vocab_[DEFAULT_EOS_STR]);
  unkId_ = Word::fromWordIndex(vocab_[DEFAULT_UNK_STR]);
  
  // LOG(info, "eos: {}; unk: {}, <s>: {}", word2string(eosId_), word2string(unkId_), vocab_["<s>"]);

  return size();
}

// helper to add missing words to vocab_
// factorIndices has been formed up to *ex*cluding position [g].
void FactoredVocab::rCompleteVocab(std::vector<size_t>& factorIndices, size_t g) {
  // reached the end
  if (g == getNumGroups()) {
    auto word = factors2word(factorIndices);
    auto v = word.toWordIndex();
    if (!vocab_.contains(v)) // add if missing
      vocab_.add(word2string(word), v);
    return;
  }
  // try next factor
  if (g == 0 || lemmaHasFactorGroup(factorIndices[0], g)) {
    for (size_t g1 = 0; g1 < factorShape_[g] - 1; g1++) {
      factorIndices[g] = g1;
      rCompleteVocab(factorIndices, g + 1);
    }
  }
  else {
    factorIndices[g] = FACTOR_NOT_APPLICABLE;
    rCompleteVocab(factorIndices, g + 1);
  }
}

size_t FactoredVocab::lemmaSize() const {
  return lemmaSize_;
}

void FactoredVocab::constructGroupInfoFromFactorVocab() {
  // form groups
  size_t numGroups = groupPrefixes_.size();
  size_t factorVocabSize = this->factorVocabSize();
  factorGroups_.resize(factorVocabSize, 0);
  for (size_t g = 1; g < groupPrefixes_.size(); g++) { // set group labels; what does not match any prefix will stay in group 0
    const auto& groupPrefix = groupPrefixes_[g];
    for (WordIndex u = 0; u < factorVocabSize; u++)
      if (utils::beginsWith(factorVocab_[u], groupPrefix)) {
        //ABORT_IF(factorGroups_[u] != 0, "Factor {} matches multiple groups, incl. {}", factorVocab_[u], groupPrefix);
        if(factorGroups_[u] != 0)
          LOG(info, "Factor {} matches multiple groups, incl. {}, using {}", factorVocab_[u], groupPrefixes_[factorGroups_[u]], groupPrefix);
        factorGroups_[u] = g;
      }
  }
  // determine group index ranges
  groupRanges_.resize(numGroups, { SIZE_MAX, (size_t)0 });
  std::vector<int> groupCounts(numGroups, 0); // number of group members
  for (WordIndex u = 0; u < factorVocabSize; u++) { // determine ranges; these must be non-overlapping, verified via groupCounts
    auto g = factorGroups_[u];
    if (groupRanges_[g].first > u)
        groupRanges_[g].first = u;
    if (groupRanges_[g].second < u + 1)
        groupRanges_[g].second = u + 1;
    groupCounts[g]++;
  }

  // required by LSH shortlist. Factored segmenter encodes the number of lemmas in the first factor group, this corresponds to actual surface forms
  lemmaSize_ = groupCounts[0];
  
  for (size_t g = 0; g < numGroups; g++) { // detect non-overlapping groups
    LOG(info, "[vocab] Factor group '{}' has {} members", groupPrefixes_[g], groupCounts[g]);
    if (groupCounts[g] == 0) { // factor group is unused  --@TODO: once this is not hard-coded, this is an error condition
      groupRanges_[g].first = g > 0 ? groupRanges_[g-1].second : 0; // fix up the entry
      groupRanges_[g].second = groupRanges_[g].first;
      continue;
    }
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
  ABORT_IF((WordIndex)virtualVocabSize() != virtualVocabSize(),
      "Too many factors, virtual index space {} exceeds the bit limit of WordIndex type", utils::withCommas(virtualVocabSize()));
}

// encode factors into a Word struct
// inputs:
//  - factorIndices[factorType] = factorIndex (e.g. 0 for |ca )
// output:
//  - representation as 'Word' (which is, in fact, a single big integer)
Word FactoredVocab::factors2word(const std::vector<size_t>& factorIndices /* [numGroups] */) const {
  size_t index = 0;
  size_t numGroups = getNumGroups();
  ABORT_IF(factorIndices.size() != numGroups, "Factor indices array size must be same as number of factor groups");
  for (size_t g = 0; g < numGroups; g++) {
    auto factorIndex = factorIndices[g];
    if (factorIndex != FACTOR_NOT_SPECIFIED) { // check validity
      auto factor0Index = factorIndices[0];    // lemma
      ABORT_IF(factor0Index == FACTOR_NOT_SPECIFIED, "Without lemma, no other factor may be specified");
      ABORT_IF(lemmaHasFactorGroup(factor0Index, g) == (factorIndex == FACTOR_NOT_APPLICABLE),
               "Lemma '{}' {} factor group '{}'",
               factorVocab_[WordIndex(factor0Index + groupRanges_[0].first)],
               lemmaHasFactorGroup(factor0Index, g) ? "needs" : "does not have",
               groupPrefixes_[g]);
    }
    if (factorIndex == FACTOR_NOT_APPLICABLE || factorIndex == FACTOR_NOT_SPECIFIED)
      factorIndex = (size_t)factorShape_[g] - 1; // sentinel for "unused" or "not specified"
    else
      ABORT_IF(factorIndex >= (size_t)factorShape_[g] - 1, "Factor index out of range");
    index += factorIndex * factorStrides_[g];
  }
  return Word::fromWordIndex(index);
}

// encode only a lemma into a 'Word'
// The result is incomplete, in that the lemma likely has additional factors that are not yet specified.
// Those are encoded as the value FACTOR_NOT_SPECIFIED. This function is used during beam search,
// which starts with lemma scores, and then adds factors one by one to the path score.
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

// factor unit: index of factor name in the joint factor vocabulary
// factor index: relative index within factor type, e.g. 0 for |ca
size_t FactoredVocab::factorUnit2FactorIndex(WordIndex u) const {
  auto g = factorGroups_[u]; // convert u to relative u within factor group range
  ABORT_IF(u < groupRanges_[g].first || u >= groupRanges_[g].second, "Invalid factorGroups_ entry??");
  return u - groupRanges_[g].first;
}

// split the 'Word' representation, which is really a single big integer, into the individual
// factor indices for all factor types
void FactoredVocab::word2factors(Word word, std::vector<size_t>& factorIndices /* [numGroups] */) const {
  size_t numGroups = getNumGroups();
  factorIndices.resize(numGroups);
  for (size_t g = 0; g < numGroups; g++) {
    auto factorIndex = getFactor(word, g);
    factorIndices[g] = factorIndex;
  }
#if 1
  auto test = factors2word(factorIndices);
  ABORT_IF(test != word, "Word <-> factor conversion broken?? {} vs{}, '{}' vs. '{}'",
           test.toWordIndex(), word.toWordIndex(), word2string(test), word2string(word));
#endif
}

// serialize 'Word' representation into its string form
std::string FactoredVocab::word2string(Word word) const {
  // this function has some code dup, so that we can bypass some checks for debugging
  size_t numGroups = getNumGroups();
  size_t factor0Index = word.toWordIndex() / factorStrides_[0];
  std::string res;
  for (size_t g = 0; g < numGroups; g++) {
    size_t index = word.toWordIndex();
    index = index / factorStrides_[g];
    index = index % (size_t)factorShape_[g];
    if (index == (size_t)factorShape_[g] - 1) { // special sentinel value for unspecified or not-applicable
      if (factor0Index >= (size_t)factorShape_[0])
        res.append("(lemma oob)");
      else if (lemmaHasFactorGroup(factor0Index, g))
        res.append("?");
    }
    else
      res.append(getFactorName(g, index));
  }
  return res;
}

// deserialize factored string form (e.g. HELLO|ci|wb) into its internal binary 'Word' representation
Word FactoredVocab::string2word(const std::string& w) const {
  auto sep = std::string(1, factorSeparator_);
  auto parts = utils::splitAny(w, sep);
  auto na = FACTOR_NOT_APPLICABLE; // (gcc compiler bug: sometimes it cannot find this if passed directly)
  std::vector<size_t> factorIndices(groupRanges_.size(), na); // default for unused factors
  for (size_t i = 0; i < parts.size(); i++) {
    WordIndex u;
    bool found = factorVocab_.tryFind(i == 0 ? parts[i] : sep + parts[i], u);
    if (!found) {
      static int logs = 5;
      if (logs > 0) {
        logs--;
        LOG(info, "WARNING: Unknown factor '{}' in '{}'; mapping to '{}'", parts[i], w, word2string(getUnkId()));
      }
      return getUnkId();
    }
    // convert u to relative u within factor group range
    auto g = factorGroups_[u];
    ABORT_IF(u < groupRanges_[g].first || u >= groupRanges_[g].second, "Invalid factorGroups_ entry??");
    factorIndices[g] = u - groupRanges_[g].first;
  }
  auto word = factors2word(factorIndices);
  return word;
}

// does a specific factor exist in the vocabulary
// Factor name must be given without separator. This function cannot be used for lemmas.
bool FactoredVocab::tryGetFactor(const std::string& factorName, size_t& groupIndex, size_t& factorIndex) const {
  WordIndex u;
  if (factorVocab_.tryFind(factorSeparator_ + factorName, u))
  {
      groupIndex = factorGroups_[u];
      ABORT_IF(u < groupRanges_[groupIndex].first || u >= groupRanges_[groupIndex].second, "Invalid factorGroups_ entry??");
      factorIndex = u - groupRanges_[groupIndex].first;
      return true;
  }
  else
      return false;
}

// extract the factor index of a given factor type from the 'Word' representation
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

#ifdef FACTOR_FULL_EXPANSION
void FactoredVocab::constructNormalizationInfoForVocab() {
  // create mappings needed for normalization in factored outputs
  //size_t numGroups = groupPrefixes_.size();
  size_t vocabSize = virtualVocabSize();
  //factorMasks_  .resize(numGroups, std::vector<float>(vocabSize, 0));     // [g][v] 1.0 if word v has factor g
  //factorIndices_.resize(numGroups, std::vector<IndexType>(vocabSize, 0)); // [g][v] index of factor (or any valid index if it does not have it; we use 0)
  gapLogMask_.resize(vocabSize, -1e8f);
  for (WordIndex v = 0; v < vocabSize; v++) {
#if 1 // @TODO: TEST THIS again by disabling factored decoding in beam_search.h
    if (vocab_.contains(v))
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
#endif

/*virtual*/ Word FactoredVocab::operator[](const std::string& word) const /*override final*/ {
  // @TODO: do away with vocab_ altogether, and just always parse.
  WordIndex index;
  bool found = vocab_.tryFind(word, index);
  if (found)
    return Word::fromWordIndex(index);
  else
    return string2word(word);
}

/*virtual*/ const std::string& FactoredVocab::operator[](Word word) const /*override final*/ {
  //LOG(info, "Looking up Word {}={}", word.toWordIndex(), word2string(word));
  ABORT_IF(!vocab_.contains(word.toWordIndex()), "Invalid factor combination {}", word2string(word));
  return vocab_[word.toWordIndex()];
}

// convert a string representation of a token sequence to all-caps by changing all capitalization factors to |ca
/*virtual*/ std::string FactoredVocab::toUpper(const std::string& line) const /*override final*/ {
  return utils::findReplace(utils::findReplace(utils::findReplace(utils::findReplace(utils::findReplace(line, "|scl", "|scu", /*all=*/true), "|ci", "|ca", /*all=*/true), "|cn", "|ca", /*all=*/true), "@CI", "@CA", /*all=*/true), "@CN", "@CA", /*all=*/true);
}

// convert a string representation of a token sequence to English title case by changing the capitalization factors to |ci
/*virtual*/ std::string FactoredVocab::toEnglishTitleCase(const std::string& line) const /*override final*/ {
  // @BUGBUG: does not handle the special words that should remain lower-case
  // note: this presently supports both @WB and @GL- (legacy)
  return utils::findReplace(utils::findReplace(utils::findReplace(utils::findReplace(utils::findReplace(line, "|scl", "|scu", /*all=*/true), "|cn|wb", "|ci|wb", /*all=*/true), "|cn|gl-", "|ci|gl-", /*all=*/true), "@CN@WB", "@CI@WB", /*all=*/true), "@CN@GL-", "@CI@GL-", /*all=*/true);
}

// convert word indices to indices of shortlist items
// We only shortlist the lemmas, hence return the lemma index (offset to correctly index into the concatenated W matrix).
// This strange pointer-based interface is for ease of interaction with our production environment.
/*virtual*/ void FactoredVocab::transcodeToShortlistInPlace(WordIndex* ptr, size_t num) const {
  for (; num-- > 0; ptr++) {
    auto word = Word::fromWordIndex(*ptr);
    auto lemmaIndex = getFactor(word, 0) + groupRanges_[0].first;
    *ptr = (WordIndex)lemmaIndex;
  }
}

// generate a valid random factored word (used by collectStats())
/*virtual*/ Word FactoredVocab::randWord() const /*override final*/ {
  auto numGroups = getNumGroups();
  std::vector<size_t> factorIndices; factorIndices.reserve(numGroups);
  for (size_t g = 0; g < numGroups; g++) {
    size_t factorIndex;
    if (g == 0 || lemmaHasFactorGroup(factorIndices[0], g))
      factorIndex = rand() % (factorShape_[g] - 1);
    else
      factorIndex = FACTOR_NOT_APPLICABLE;
    factorIndices.push_back(factorIndex);
  }
  return factors2word(factorIndices);
}

// encode a string representation of an entire token sequence, as found in the corpus file, into a 'Word' array
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

// decode a 'Word' array into the external string representation of that token sequence, as written to output files
/*virtual*/ std::string FactoredVocab::decode(const Words& sentence, bool ignoreEOS /*= true*/) const /*override final*/ {
  std::vector<std::string> decoded; decoded.reserve(sentence.size());
  for(auto w : sentence)
    if((w != getEosId() || !ignoreEOS))
      decoded.push_back((*this)[w]);
  return utils::join(decoded, " ");
}

// diagnostics version of decode() that will not fail on partial words, will print EOS, and is a little slower
std::string FactoredVocab::decodeForDiagnostics(const Words& sentence) const {
  std::vector<std::string> decoded; decoded.reserve(sentence.size());
  for (auto w : sentence)
    decoded.push_back(word2string(w));
  return utils::join(decoded, " ");
}

// helper to unescape \x.. and \u....
static void unescapeHexEscapes(std::string& utf8Lemma) {
  if (utf8Lemma.find('\\') == std::string::npos)
    return; // nothing to do
  auto lemma = utils::utf8ToUtf16String(utf8Lemma); // \u.... implies we must operate on UTF-16 level (not UCS-4)
  auto pos = lemma.find('\\');
  while (pos != std::string::npos) {
    ABORT_IF(pos + 1 >= lemma.size() || (lemma[pos+1] != 'x' && lemma[pos + 1] != 'u'), "Malformed escape in factored encoding: {}", utf8Lemma);
    int numDigits = 2 + 2 * (lemma[pos + 1] == 'u'); // 2 for \x, 4 for \u
    ABORT_IF(pos + 2 + numDigits > lemma.size(), "Malformed escape in factored encoding: {}", utf8Lemma);
    auto digits = utils::utf8FromUtf16String(lemma.substr(pos + 2, numDigits));
    auto c = std::strtoul(digits.c_str(), nullptr, 16);
    lemma[pos] = (char16_t)c;
    lemma.erase(pos + 1, 1 + numDigits);
    pos = lemma.find('\\', pos+1);
  }
  utf8Lemma = utils::utf8FromUtf16String(lemma);
}

// convert a 'Word' sequence to its final human-readable surface form
// This interprets the capitalization and glue factors.
// This assumes a specific notation of factors, emulating our C# code for generating these factors:
//  - | as separator symbol
//  - capitalization factors are cn, ci, and ca
//  - glue factors are gl+, gr+, wbn, wen, cbn, cen
std::string FactoredVocab::surfaceForm(const Words& sentence) const /*override final*/ {
  std::string res;
  res.reserve(sentence.size() * 10);
  bool prevHadGlueRight = true; // no space at sentence start
  for(auto w : sentence) {
    if (w == getEosId())
      break;
    auto token = (*this)[w];
    auto tokens = utils::split(token, "|");
    //std::cerr << token << " ";
    auto lemma = tokens[0];
    std::set<std::string> tokenSet(tokens.begin() + 1, tokens.end());
    auto has = [&](const char* factor) { return tokenSet.find(factor) != tokenSet.end(); };
    // spacing
    bool hasGlueRight = has("gr+") || has("wen") || has("cen");
    bool hasGlueLeft  = has("gl+") || has("wbn") || has("cbn") || has("wi");
    bool insertSpaceBefore = !prevHadGlueRight && !hasGlueLeft;
    if (insertSpaceBefore)
      res.push_back(' ');
    prevHadGlueRight = hasGlueRight;
    // capitalization
    unescapeHexEscapes(lemma); // unescape \x.. and \u....
    if (utils::beginsWith(lemma, "\xE2\x96\x81"))  // remove leading _ (\u2581, for DistinguishInitialAndInternalPieces mode)
        lemma = lemma.substr(3);
    if      (has("ci"))  lemma = utils::utf8Capitalized(lemma);
    else if (has("ca"))  lemma = utils::utf8ToUpper    (lemma);
    else if (has("cn"))  lemma = utils::utf8ToLower    (lemma);
    else if (has("scu")) lemma = utils::utf8ToUpper    (lemma);
    else if (has("scl")) lemma = utils::utf8ToLower    (lemma);
    res.append(lemma);
  }
  //std::cerr << "\n" << res << "\n";
  return res;
}

/**
 * Auxiliary function that return the total number of factors (no lemmas) in a factored vocabulary.
 * @return number of factors
 */
size_t FactoredVocab::getTotalFactorCount() const {
  return factorVocabSize() - groupRanges_[0].second;
}

/**
 * Decodes the indexes of lemma and factor for each word and outputs that information separately.
 * It will return two data structures that contain separate information regarding lemmas and factors indexes
 * by receiving a list with the word indexes of a batch.
 * @param[in] words           vector of words
 * @param[out] lemmaIndices   lemma index for each word
 * @param[out] factorIndices  factor usage information for each word (1 if the factor is used 0 if not)
 */
void FactoredVocab::lemmaAndFactorsIndexes(const Words& words, std::vector<IndexType>& lemmaIndices, std::vector<float>& factorIndices) const {
  lemmaIndices.reserve(words.size());
  factorIndices.reserve(words.size() * getTotalFactorCount());

  auto numGroups = getNumGroups();
  std::vector<size_t> lemmaAndFactorIndices;

  for (auto &word : words) {
    if (vocab_.contains(word.toWordIndex())) { // skip invalid combinations in the space (can only happen during initialization)  --@TODO: add a check?
      word2factors(word, lemmaAndFactorIndices);
      lemmaIndices.push_back((IndexType) lemmaAndFactorIndices[0]); // save the lemma vocabulary index
      for (size_t g = 1; g < numGroups; g++) { // loop over the different factors group
        auto factorIndex = lemmaAndFactorIndices[g]; // get the vocabulary index of the factor of group g
        ABORT_IF(factorIndex == FACTOR_NOT_SPECIFIED, "Attempted to embed a word with a factor not specified");
        for (int i = 0; i < factorShape_[g] - 1; i++) { // loop over all factors in group g
          factorIndices.push_back((float) (factorIndex == i)); // fill the factor indexes array with '0' if the factor is not used in a given word, '1' if it is
        }
      }
    }
  }
}

// create a CSR matrix M[V,U] from words[] with M[v,u] = 1 if factor u is a factor of word v
// This is used to form the embedding of a multi-factor token.
// That embedding is a sum of the embeddings of the individual factors.
// Those individual embeddings are assumed to be concatenated into one joint large embedding matrix.
// The factor embeddings are summed up by multiplying the joint embedding matrix with a sparse matrix
// that contains a 1 for all positions in the joint matrix that should be summed up.
// This function creates that sparse matrix in CSR form.
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
    if (vocab_.contains(word.toWordIndex())) { // skip invalid combinations in the space (can only happen during initialization)  --@TODO: add a check?
      word2factors(word, factorIndices);
      for (size_t g = 0; g < numGroups; g++) { // @TODO: make this faster by having a list of all factors to consider for a lemma?
        auto factorIndex = factorIndices[g];
        ABORT_IF(factorIndex == FACTOR_NOT_SPECIFIED, "Attempted to embed a word with a factor not specified");
        if (factorIndex == FACTOR_NOT_APPLICABLE)
          continue;
        indices.push_back((IndexType)(factorIndex + groupRanges_[g].first)); // map to unit index
        weights.push_back(1.0f);
      }
    }
    offsets.push_back((IndexType)indices.size()); // next matrix row begins at this offset
  }
  return { Shape({(int)words.size(), (int)factorVocabSize()}), weights, indices, offsets };
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
  ABORT_IF(!wasInserted, "Duplicate vocab entry for '{}', new index {} vs. existing index {}", word, index, str2index_[word]);
  wasInserted = index2str_.insert(std::make_pair(index, word)).second;
  ABORT_IF(!wasInserted, "Duplicate vocab entry for index {} (new: '{}'; existing: '{}')", index, word, index2str_[index]);
  return index;
}

static const std::string g_emptyString;
const std::string& FactoredVocab::WordLUT::operator[](WordIndex index) const {
  auto iter = index2str_.find(index);
  if (iter == index2str_.end())
    // returns an empty string for unknown index values
    // @TODO: is that ever used ? If so, document.If not, remove this feature and let it fail.static const std::string g_emptyString;
    return g_emptyString; // (using a global since we return a reference)
  else
    return iter->second;
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
size_t FactoredVocab::WordLUT::load(const std::string& path) {
  std::string line;
  io::InputFileStream in(path);
  for (WordIndex v = 0; io::getline(in, line); v++)
    add(line, v);
  return size();
}

void FactoredVocab::WordLUT::dumpToFile(const std::string& path) {
  io::OutputFileStream out(path);
  for (auto kvp : index2str_)
    out << kvp.second << "\t" << utils::withCommas(kvp.first) << "\n";
}

const static std::vector<std::string> exts{ ".fsv", ".fm"/*legacy*/ }; // @TODO: delete the legacy one

// Note: This does not actually load it, only checks the path for the type.
// Since loading takes a while, we cache instances.
Ptr<IVocab> createFactoredVocab(const std::string& vocabPath) {
  // this can be multi-threaded, so must run under lock
  static std::mutex s_mtx;
  std::lock_guard<std::mutex> criticalSection(s_mtx);

  bool isFactoredVocab = std::any_of(exts.begin(), exts.end(), [&](const std::string& ext) { return utils::endsWith(vocabPath, ext); });
  if (isFactoredVocab) {
    static std::map<std::string, Ptr<IVocab>> s_cache;
    auto iter = s_cache.find(vocabPath);
    if (iter != s_cache.end()) {
      LOG_ONCE(info, "[vocab] Reusing existing vocabulary object in memory (vocab size {})", iter->second->size());
      return iter->second;
    }
    auto vocab = New<FactoredVocab>();
    s_cache.insert(std::make_pair(vocabPath, vocab));
    return vocab;
  }
  else
    return nullptr;
}
/*virtual*/ const std::vector<std::string>& FactoredVocab::suffixes() const /*override final*/ {
  return exts;
}

}  // namespace marian
