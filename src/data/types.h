#pragma once

#include "common/definitions.h"

#include <cstdint>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>

namespace marian {

// Type for all vocabulary items, based on IndexType
typedef IndexType WordIndex;    // WordIndex is used for words or tokens arranged in consecutive order
class Word {                    // Word is an abstraction of a unique id, not necessarily consecutive
  WordIndex wordId_;
public:
  // back compat with WordIndex
  Word(std::size_t wordId) : wordId_((WordIndex)wordId) {} // @TODO: make explicit, or make private
  operator WordIndex() const { return getWordIndex(); }

  // needed for STL containers
  Word() : wordId_((WordIndex)-1) {}
  bool operator==(const Word& other) const { return wordId_ == other.wordId_; }
  std::size_t hash() const { return std::hash<WordIndex>{}(wordId_); }

  // main methods and constants
  static Word From(std::size_t wordId) { return Word(wordId); }
  const WordIndex& getWordIndex() const { return wordId_; }

  static Word NONE; // @TODO: decide whether we need this, in additional Word()
  // EOS and UNK are placed in these positions in Marian-generated vocabs
  static Word DEFAULT_EOS_ID;
  static Word DEFAULT_UNK_ID;
};

// Sequence of vocabulary items
typedef std::vector<Word> Words;

// Helper to map a Word vector to a WordIndex vector
static inline std::vector<WordIndex> toWordIndexVector(const Words& words) {
  return std::vector<WordIndex>(words.begin(), words.end());
}

// names of EOS and UNK symbols
const std::string DEFAULT_EOS_STR = "</s>";
const std::string DEFAULT_UNK_STR = "<unk>";

// alternatively accepted names in Yaml dictionaries for ids 0 and 1, resp.
const std::string NEMATUS_EOS_STR = "eos";
const std::string NEMATUS_UNK_STR = "UNK";

}  // namespace marian

namespace std {
  template<> struct hash<marian::Word> {
    std::size_t  operator()(const marian::Word& s) const noexcept { return s.hash(); }
  };
}
