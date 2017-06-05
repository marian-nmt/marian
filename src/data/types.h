#pragma once

#include <cstdint>
#include <cstdlib>
#include <unordered_map>
#include <vector>

namespace marian {

typedef size_t Word;
typedef std::vector<Word> Words;

const Word EOS_ID = 0;
const Word UNK_ID = 1;
const std::string EOS_STR = "</s>";
const std::string UNK_STR = "<unk>";

const Word STP_ID = 2;
const Word CPY_ID = 3;
const Word DEL_ID = 4;
const Word RPL_ID = 5;

const std::string STP_STR = "<step>";
const std::string CPY_STR = "<c>";
const std::string DEL_STR = "<d>";
const std::string RPL_STR = "<r>";

const std::unordered_map<std::string, Word> SPEC2SYM = {
    {STP_STR, STP_ID}, {CPY_STR, CPY_ID}, {DEL_STR, DEL_ID}, {RPL_STR, RPL_ID},
};

const std::unordered_map<Word, std::string> SYM2SPEC = {
    {STP_ID, STP_STR}, {CPY_ID, CPY_STR}, {DEL_ID, DEL_STR}, {RPL_ID, RPL_STR},
};
}