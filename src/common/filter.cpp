#include "common/filter.h"

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <cmath>
#include <algorithm>

#include "decoder/god.h"
#include "common/vocab.h"
#include "common/utils.h"
#include "common/types.h"

Filter::Filter(const size_t numFirstWords) : numFirstWords_(numFirstWords) {}

Filter::Filter(const Vocab& srcVocab,
               const Vocab& trgVocab,
               const std::string& path,
               const size_t numFirstWords,
               const size_t maxNumTranslation)
  : numFirstWords_(numFirstWords),
    mapper_(ParseAlignmentFile(srcVocab,
                               trgVocab,
                               path,
                               maxNumTranslation,
                               numFirstWords)) {}

std::vector<Words> Filter::ParseAlignmentFile(const Vocab& srcVocab,
                                              const Vocab& trgVocab,
                                              const std::string& path,
                                              const size_t maxNumTranslation,
                                              const size_t numNFirst) {
  std::map<Word, std::vector<std::pair<Word, float>>> mapper;
  std::ifstream filterFile(path);
  std::string line;
  while (std::getline(filterFile, line)) {
    Trim(line);
    if (line.size() == 0) {
      continue;
    }
    std::vector<std::string> tokens;
    Split(line, tokens, " ");
    if (tokens.size() != 3) {
      LOG(info) << "Filter: broken line: " << line;
      continue;
    }
    if (trgVocab[tokens[0]] != UNK && srcVocab[tokens[1]] != UNK) {
      mapper[srcVocab[tokens[1]]].push_back(std::make_pair(trgVocab[tokens[0]],
                                                           std::stof(tokens[2])));
    }
  }
  std::vector<Words> vecMapper(srcVocab.size());
  for (size_t i = 0; i < srcVocab.size(); ++i) {
    if (mapper.find(i) != mapper.end()) {
      std::sort(mapper[i].begin(), mapper[i].end(),
          [](const std::pair<Word, float>& left,
            const std::pair<Word, float>& right) {
            return left.second > right.second; });
      for (size_t j = 0; j < std::min(mapper[i].size(), maxNumTranslation); ++j) {
        if (mapper[i][j].first >= numNFirst) {
          vecMapper[i].push_back(mapper[i][j].first);
        }
      }
    }
  }
  return vecMapper;
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

  Words output(filtered.begin(), filtered.end());
  // std::sort(output.begin(), output.end());

  return output;
}

size_t Filter::GetNumFirstWords() const {
  return numFirstWords_;
}

void Filter::SetNumFirstWords(const size_t numFirstWords) {
  numFirstWords_ = numFirstWords;
}
