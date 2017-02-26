#pragma once

#include <cstdlib>
#include <cstdint>
#include <vector>

typedef size_t Word;
typedef std::vector<Word> Words;

const Word EOS_ID = 0;
const Word UNK_ID = 1;

const std::string EOS_STR = "</s>";
const std::string UNK_STR = "<unk>";
