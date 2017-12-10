#include "sentence.h"
#include "god.h"
#include "utils.h"
#include "common/vocab.h"

using namespace std;

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
    // fill in the factors as well so that there aren't any surprises
    // if somebody decides to look up the factors in the decoder or something
    FillDummyFactors(words_.back());
}

Sentence::Sentence(God&, size_t lineNum, const std::vector<uint>& words)
  : lineNum_(lineNum) {
    words_.push_back(words);
    // fill in the factors as well so that there aren't any surprises
    // if somebody decides to look up the factors in the decoder or something
    FillDummyFactors(words_.back());
}

void Sentence::FillDummyFactors(const Words& line) {
  factors_.emplace_back(FactWords(line.size(), FactWord(1)));
  FactWords& factline = factors_.back();
  for (size_t i = 0; i < line.size(); ++i) {
    factline[i][0] = line[i];
  }
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

