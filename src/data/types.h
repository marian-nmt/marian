#pragma once

#include "common/definitions.h"

#include <cstdint>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>

namespace marian {

// Type for all vocabulary items, based on IndexType
typedef IndexType Word;

// Sequence of vocabulary items
typedef std::vector<Word> Words;

// EOS and UNK are placed in these positions in Marian-generated vocabs
const Word DEFAULT_EOS_ID = 0;
const Word DEFAULT_UNK_ID = 1;

// names of EOS and UNK symbols
const std::string DEFAULT_EOS_STR = "</s>";
const std::string DEFAULT_UNK_STR = "<unk>";

// alternatively accepted names in Yaml dictionaries for ids 0 and 1, resp.
const std::string NEMATUS_EOS_STR = "eos";
const std::string NEMATUS_UNK_STR = "UNK";

}  // namespace marian
