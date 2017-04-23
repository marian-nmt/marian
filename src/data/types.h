#pragma once

#include <cstdlib>
#include <cstdint>
#include <vector>

typedef size_t Word;
typedef std::vector<Word> Words;

const Word EOS_ID = 0;
const Word UNK_ID = 1;
const Word STP_ID = 2;
const Word CPY_ID = 3;
const Word DEL_ID = 4;

const std::string EOS_STR  = "</s>";
const std::string UNK_STR  = "<unk>";
const std::string STP_STR = "<step>";
const std::string CPY_STR = "<c>";
const std::string DEL_STR = "<d>";
