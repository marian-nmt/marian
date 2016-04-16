#pragma once

#include <string>
#include <vector>
#include <memory>

#include "vocab.h"

namespace lm {
  namespace ngram {
    class ProbingModel;
    class State;
  }
  
  typedef unsigned int WordIndex;
}

class KenlmState {
  private:
    lm::ngram::State* state_;
  
  public:
    KenlmState();
    KenlmState(const KenlmState&);
    KenlmState& operator=(const KenlmState&);
    ~KenlmState();
    
    KenlmState(KenlmState&&) = delete;
    
    
    lm::ngram::State& operator*();
    lm::ngram::State& operator*() const;
    
    bool operator==(const KenlmState&);
    
    friend uint64_t hash_value(const KenlmState&);

};

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
    void BeginSentenceState(KenlmState&) const;
    WordPairs::const_iterator begin() const;
    WordPairs::const_iterator end() const;
    size_t GetIndex() const;
    float GetWeight() const;
    
  private:
    std::unique_ptr<KenlmModel> lm_;
    WordPairs vm_;
    size_t index_;
    float weight_;
};
