#include "kenlm.h"
#include "lm/model.hh"
    
class VocabGetter : public lm::EnumerateVocab {
  public:
    VocabGetter(WordPairs& vm, const Vocab& vocab)
    : vm_(vm), vocab_(vocab)
    {
      vm_.emplace_back(2, EOS); // is there a constant for "</s>" = 2?
      vm_.emplace_back(1, UNK); // is there a constant for "<s>" = 1?
      vm_.emplace_back(lm::kUNK, UNK);
    }
    
    void Add(lm::WordIndex index, const StringPiece &str) {
      size_t word = vocab_[str.as_string()];
      if(word > 2)
        vm_.emplace_back(index, word);
    }
    
  private:
    WordPairs& vm_;
    const Vocab& vocab_;
};
    
LM::LM(const std::string& path, const Vocab& vocab) {
  lm::ngram::Config config;
  VocabGetter* vg = new VocabGetter(vm_, vocab);
  config.enumerate_vocab = vg;
  lm_.reset(new KenlmModel(path.c_str(), config));
  delete vg;
}

LM::~LM() {}

LM::LM(LM&& lm)
 : lm_(std::move(lm.lm_)), vm_(std::move(lm.vm_))
{}

float LM::Score(const KenlmState& in, lm::WordIndex index, KenlmState& out) const {
  float cost = lm_->FullScore(in, index, out).prob;
  return cost;
}

const KenlmState& LM::BeginSentenceState() const {
  return lm_->BeginSentenceState();
}

WordPairs::const_iterator LM::begin() const {
  return vm_.begin();    
}

WordPairs::const_iterator LM::end() const {
  return vm_.end();    
}

size_t LM::size() const {
  return vm_.size();
}
