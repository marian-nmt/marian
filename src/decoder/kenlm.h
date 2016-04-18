#pragma once

#include <string>
#include <vector>
#include <memory>

#include "vocab.h"

#include "lm/state.hh"

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
    LM(const std::string& path, const Vocab& vocab, size_t index, float weight);
    LM(LM&& lm);
    ~LM();
    
    float Score(const KenlmState& in, lm::WordIndex index, KenlmState& out) const;
    const KenlmState& BeginSentenceState() const;
    WordPairs::const_iterator begin() const;
    WordPairs::const_iterator end() const;
    size_t GetIndex() const;
    float GetWeight() const;
    size_t size() const;
    
  private:
    std::unique_ptr<KenlmModel> lm_;
    WordPairs vm_;
    size_t index_;
    float weight_;
};
