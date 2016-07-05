#include "common/filter.h"

#include <string>
#include <vector>
#include <fstream>
#include <memory>
#include <set>
#include <cmath>
#include <algorithm>

#include "decoder/god.h"
#include "common/vocab.h"
#include "common/utils.h"
#include "common/types.h"

Filter::Filter(const size_t numFirstWords) : numFirstWords_(numFirstWords) {}

Filter::Filter(const std::string& path, const size_t numFirstWords)
  : numFirstWords_(numFirstWords),
    mapper_(ParseAlignmentFile(path)) {}

std::vector<Words> Filter::ParseAlignmentFile(const std::string& path) {
  std::vector<Words> mapper;
  std::fstream filterFile(path);
  std::string line;
  Vocab& trgVocab = God::GetTargetVocab();
  while (std::getline(filterFile, line)) {
    Trim(line);
    if (line.size() == 0) {
      mapper.push_back(Words());
      continue;
    }
    std::vector<std::string> tokens;
    Split(line, tokens);
    Words words;
    for (auto& token : tokens) {
      words.push_back(trgVocab[token]);
    }

    mapper.push_back(words);
  }
  return mapper;
}

Words Filter::GetFilteredVocab(const Words& srcWords, const size_t maxVocabSize) const {
  std::set<Word> filtered;

  for(size_t i = 0; i < std::min(numFirstWords_, maxVocabSize); ++i) {
    filtered.insert(i);
  }

  for (const auto& srcWord : srcWords) {
    for (const auto& trgWord : mapper_[srcWord]) {
      if (trgWord < maxVocabSize) {
        filtered.insert(trgWord);
      }
    }
  }

  Words output(filtered.cbegin(), filtered.cend());
  std::sort(output.begin(), output.end());

  return output;
}

size_t Filter::GetNumFirstWords() const {
  return numFirstWords_;
}

void Filter::SetNumFirstWords(const size_t numFirstWords) {
  numFirstWords_ = numFirstWords;
}
