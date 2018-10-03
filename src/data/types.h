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

const Word STP_ID = 2;
const Word CPY_ID = 3;
const Word DEL_ID = 4;
const Word RPL_ID = 5;

const std::string STP_STR = "<step>";
const std::string CPY_STR = "<c>";
const std::string DEL_STR = "<d>";
const std::string RPL_STR = "<r>";

const std::unordered_map<std::string, Word> SPEC2SYM = {
    {STP_STR, STP_ID},
    {CPY_STR, CPY_ID},
    {DEL_STR, DEL_ID},
    {RPL_STR, RPL_ID},
};

const std::unordered_map<Word, std::string> SYM2SPEC = {
    {STP_ID, STP_STR},
    {CPY_ID, CPY_STR},
    {DEL_ID, DEL_STR},
    {RPL_ID, RPL_STR},
};
}  // namespace marian
