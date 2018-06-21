#include <sstream>
#include "sentence.h"
#include "god.h"
#include "utils.h"
#include "common/vocab.h"

using namespace std;

namespace amunmt {

Sentence::Sentence(const God &god, unsigned vLineNum, const std::string& line)
  : lineNum_(vLineNum)
{
  std::vector<std::string> tabs;
  Split(line, tabs, "\t");
  if (tabs.size() == 0) {
    tabs.push_back("");
  }

  unsigned maxLength = god.Get<unsigned>("max-length");
  unsigned i = 0;
  for (auto& tab : tabs) {
    std::vector<std::string> lineTokens;
    Trim(tab);
    Split(tab, lineTokens, " ");

    if (lineTokens.size() == 0) {
      // empty line
      words_.emplace_back(Words());
      factors_.emplace_back(FactWords());
    }
    else {
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
      for (unsigned i = 0; i < factors_.back().size(); ++i) {
        lineWords[i] = factors_.back()[i][0];
      }
      words_.emplace_back(lineWords);
    }
    i++;
  }
}

Sentence::Sentence(const God &god, unsigned lineNum, const std::vector<std::string>& words)
  : lineNum_(lineNum) {
    auto processed = god.Preprocess(0, words);
    words_.push_back(god.GetSourceVocab(0)(processed));
    // fill in the factors as well so that there aren't any surprises
    // if somebody decides to look up the factors in the decoder or something
    FillDummyFactors(words_.back());
}

Sentence::Sentence(God&, unsigned lineNum, const std::vector<unsigned>& words)
  : lineNum_(lineNum) {
    words_.push_back(words);
    // fill in the factors as well so that there aren't any surprises
    // if somebody decides to look up the factors in the decoder or something
    FillDummyFactors(words_.back());
}

void Sentence::FillDummyFactors(const Words& line) {
  factors_.emplace_back(FactWords(line.size(), FactWord(1)));
  FactWords& factline = factors_.back();
  for (unsigned i = 0; i < line.size(); ++i) {
    factline[i][0] = line[i];
  }
}

unsigned Sentence::GetLineNum() const {
  return lineNum_;
}

const Words& Sentence::GetWords(unsigned index) const {
  return words_[index];
}

const FactWords& Sentence::GetFactors(unsigned index) const {
  return factors_[index];
}

unsigned Sentence::size(unsigned index) const {
  return words_[index].size();
}

std::string Sentence::Debug(unsigned verbosity) const
{
  const FactWords &words = factors_[0];

  std::stringstream strm;
  for (unsigned i = 0; i < words.size(); ++i) {
    const FactWord &word = words[i];
    strm << word[0] << " ";
  }
  return strm.str();
}

}

