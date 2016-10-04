#include "sentence.h"
#include "god.h"
#include "utils.h"
#include "common/vocab.h"

Sentence::Sentence(size_t lineNo, const std::string& line)
: lineNo_(lineNo), line_(line)
{
  std::vector<std::string> tabs;
  Split(line, tabs, "\t");
  size_t i = 0;
  for(auto&& tab : tabs) {
    std::vector<std::string> lineTokens;
    Trim(tab);
    Split(tab, lineTokens, " ");
    auto processed = God::Preprocess(i, lineTokens);
    words_.push_back(God::GetSourceVocab(i++)(processed));
  }
}

const Words& Sentence::GetWords(size_t index) const {
  return words_[index];
}

size_t Sentence::GetLine() const {
  return lineNo_;
}

