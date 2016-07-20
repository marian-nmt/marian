#include "decoder/sentence.h"

#include "common/utils.h"
#include "decoder/god.h"

Sentence::Sentence(size_t lineNo, const std::string& line)
: lineNo_(lineNo), line_(line)
{
  std::vector<std::string> tabs;
  Split(line, tabs, "\t");
  size_t i = 0;
  for(auto&& tab : tabs)
    words_.push_back(God::GetSourceVocab(i++)(tab));
}

const Words& Sentence::GetWords(size_t index) const {
  return words_[index];
}

size_t Sentence::GetLine() const {
  return lineNo_;
}

