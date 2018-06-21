#pragma once
#include <memory>
#include <vector>
#include <string>
#include "types.h"

namespace amunmt {

class God;

class Sentence {
  public:

    Sentence(const God &god, unsigned vLineNum, const std::string& line);
    Sentence(const God &god, unsigned vLineNum, const std::vector<std::string>& words);
		Sentence(God &god, unsigned lineNum, const std::vector<unsigned>& words);

    const Words& GetWords(unsigned index = 0) const;
    const FactWords& GetFactors(unsigned index = 0) const;
    unsigned size(unsigned index = 0) const;

    unsigned GetLineNum() const;

    std::string Debug(unsigned verbosity = 1) const;

  private:
    void FillDummyFactors(const Words& line);

    std::vector<Words> words_;
    std::vector<FactWords> factors_;
    unsigned lineNum_;

    Sentence(const Sentence &) = delete;
};

using SentencePtr = std::shared_ptr<Sentence>;



}
