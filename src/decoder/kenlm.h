#pragma once

#include <string>
#include <vector>
#include <memory>

#include "vocab.h"

#pragma diag_suppress 172
#include "lm/state.hh"
#pragma diag_default 172

namespace lm {
  namespace ngram {
    class ProbingModel;
  }
  
  typedef unsigned int WordIndex;
}

typedef lm::ngram::State KenlmState;

typedef std::pair<lm::WordIndex, Word> WordPair;
typedef std::vector<WordPair> WordPairs;
    
class LM {
  private:
    typedef lm::ngram::ProbingModel KenlmModel;
    
  public:
    LM(const std::string& path, const Vocab& vocab);
    LM(LM&& lm);
    ~LM();
    
    float Score(const KenlmState& in, lm::WordIndex index, KenlmState& out) const;
    const KenlmState& BeginSentenceState() const;
    WordPairs::const_iterator begin() const;
    WordPairs::const_iterator end() const;
    size_t size() const;
    
  private:
    std::unique_ptr<KenlmModel> lm_;
    WordPairs vm_;
};
