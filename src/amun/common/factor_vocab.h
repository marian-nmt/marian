#pragma once

#include <string>
#include <vector>
#include <memory>
#include <algorithm>

#include "common/types.h"
#include "common/vocab.h"

namespace amunmt {
class FactorVocab {
  public:
    FactorVocab(const std::string& path);
    FactorVocab(const std::vector<std::string>& paths);
    FactWord operator[](const std::vector<std::string>& factors) const;
    FactWords operator()(const std::vector<std::vector<std::string>>& lineFactors,
                                     bool addEOS=true) const;
    Vocab& GetVocab(size_t factorIdx) const;
  private:
    typedef std::unique_ptr<Vocab> VocabPtr;
    std::vector<VocabPtr> vocabs_;
};
}
