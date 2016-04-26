#include "sentence.h"
#include "god.h"

Sentence::Sentence(size_t lineNo, const std::string& line)
: lineNo_(lineNo), line_(line)
{
  words_.push_back(God::GetSourceVocab(0)(line));
}

const Words& Sentence::GetWords(size_t index) const {
  return words_[index];
}

size_t Sentence::GetLine() const {
  return lineNo_;
}

