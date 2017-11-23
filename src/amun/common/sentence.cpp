#include "sentence.h"
#include "god.h"
#include "utils.h"
#include "common/vocab.h"

namespace amunmt {

Sentence::Sentence(const God &god, size_t vLineNum, const std::string& line)
  : lineNum_(vLineNum)
{
  std::vector<std::string> tabs;
  Split(line, tabs, "\t");
  if (tabs.size() == 0) {
    tabs.push_back("");
  }

  size_t maxLength = god.Get<size_t>("max-length");
  size_t i = 0;
  for (auto& tab : tabs) {
    std::vector<std::string> lineTokens;
    Trim(tab);
    Split(tab, lineTokens, " ");

    if (maxLength && lineTokens.size() > maxLength) {
      lineTokens.resize(maxLength);
    }

    std::vector<std::vector<std::string>> lineFactors;
    for (const std::string& token : lineTokens) {
      std::vector<std::string> wordFactors;
      Split(token, wordFactors, "|");
      lineFactors.push_back(wordFactors);
    }

    auto processed = god.Preprocess(i, lineFactors);
    factors_.emplace_back(god.GetSourceVocabs(i)(processed));
    Words lineWords(factors_.back().size());
    for (size_t i = 0; i < factors_.back().size(); ++i) {
      lineWords[i] = factors_.back()[i][0];
    }
    words_.emplace_back(lineWords);
    i++;
  }
}

Sentence::Sentence(const God &god, size_t lineNum, const std::vector<std::string>& words)
  : lineNum_(lineNum) {
    auto processed = god.Preprocess(0, words);
    words_.push_back(god.GetSourceVocab(0)(processed));
}

Sentence::Sentence(God&, size_t lineNum, const std::vector<size_t>& words)
  : lineNum_(lineNum) {
    words_.push_back(words);
}


size_t Sentence::GetLineNum() const {
  return lineNum_;
}

const Words& Sentence::GetWords(size_t index) const {
  return words_[index];
}

const FactWords& Sentence::GetFactors(size_t index) const {
  return factors_[index];
}

size_t Sentence::size(size_t index) const {
  return words_[index].size();
}


}

