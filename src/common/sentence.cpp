#include <algorithm>
#include "sentence.h"
#include "god.h"
#include "utils.h"
#include "common/vocab.h"

Sentence::Sentence(size_t vLineNum, const std::string& line)
  : lineNum_(vLineNum), line_(line)
{
  std::vector<std::string> tabs;
  Split(line, tabs, "\t");
  if (tabs.size() == 0) {
    tabs.push_back("");
  }
  size_t i = 0;
  for (auto& tab : tabs) {
    std::vector<std::string> lineTokens;
    Trim(tab);
    Split(tab, lineTokens, " ");
    auto processed = God::Preprocess(i, lineTokens);
    words_.push_back(God::GetSourceVocab(i++)(processed));
  }
}

Sentence::Sentence(size_t lineNum, const std::vector<std::string>& words)
  : lineNum_(lineNum) {
    auto processed = God::Preprocess(0, words);
    words_.push_back(God::GetSourceVocab(0)(processed));
}

size_t Sentence::GetLineNum() const {
  return lineNum_;
}

const Words& Sentence::GetWords(size_t index) const {
  return words_[index];
}

/////////////////////////////////////////////////////////
Sentences::Sentences(size_t vTaskCounter, size_t vBunchId)
  : taskCounter(vTaskCounter)
  , bunchId(vBunchId)
  , maxLength_(0)
{}

Sentences::~Sentences()
{}

void Sentences::push_back(SentencePtr sentence) {
  const Words &words = sentence->GetWords(0);
  size_t len = words.size();
  if (len > maxLength_) {
    maxLength_ = len;
  }

  coll_.push_back(sentence);
}

class LengthOrderer {
 public:
  bool operator()(const SentencePtr& a, const SentencePtr& b) const {
    return a->GetWords(0).size() < b->GetWords(0).size();
  }
};

void Sentences::SortByLength() {
  std::sort(coll_.rbegin(), coll_.rend(), LengthOrderer());
}

