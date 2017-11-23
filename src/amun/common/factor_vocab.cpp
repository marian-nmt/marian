#include "common/factor_vocab.h"

namespace amunmt {

  FactorVocab::FactorVocab(const std::string& path) {
    vocabs_.emplace_back(new Vocab(path));
  }

  FactorVocab::FactorVocab(const std::vector<std::string>& paths) {
    for (auto path : paths) {
      vocabs_.emplace_back(new Vocab(path));
    }
  }

  FactWord FactorVocab::operator[](const std::vector<std::string>& factors) const {
    FactWord factorIds(factors.size());
    for (size_t i = 0; i < factors.size(); ++i) {
      const std::string& factor = factors[i];
      factorIds[i] = (*vocabs_[i])[factor];
    }
    return factorIds;
  }

  FactWords FactorVocab::operator()(const std::vector<std::vector<std::string>>& lineFactors, bool addEOS) const {
    FactWords words(lineFactors.size());
    std::transform(lineFactors.begin(), lineFactors.end(), words.begin(),
                   [&](const std::vector<std::string>& factors) {return (*this)[factors];});
    if(addEOS)
      words.push_back(FactWord(words.back().size(), EOS_ID));
    return words;
  }

  Vocab& FactorVocab::GetVocab(size_t factorIdx) const {
    return *vocabs_.at(factorIdx);
  }
}
