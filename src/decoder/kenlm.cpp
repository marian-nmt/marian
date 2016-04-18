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
    
KenlmState::KenlmState()
: state_(new lm::ngram::State())
{}

KenlmState::KenlmState(const KenlmState& s)
: state_(new lm::ngram::State())
{
  *state_ = *s.state_;
}

KenlmState& KenlmState::operator=(const KenlmState &s) {
  *state_ = *s.state_;
  return *this;
  
}

KenlmState::~KenlmState() {
  delete state_;
}

lm::ngram::State& KenlmState::operator*() {
  return *state_;
}

lm::ngram::State& KenlmState::operator*() const {
  return *state_;
}

bool KenlmState::operator==(const KenlmState& o) {
  return *state_ == *o.state_;
}

uint64_t hash_value(const KenlmState& s) {
  //for(size_t i = 0; i < s.state_->length; i++)
  //  std::cerr << s.state_->words[i] << " ";
  return lm::ngram::hash_value(*s.state_);
}

LM::LM(const std::string& path, const Vocab& vocab, size_t index, float weight)
 : index_(index), weight_(weight) {
  lm::ngram::Config config;
  VocabGetter* vg = new VocabGetter(vm_, vocab);
  config.enumerate_vocab = vg;
  lm_.reset(new KenlmModel(path.c_str(), config));
  delete vg;
}

LM::~LM() {}

LM::LM(LM&& lm)
 : lm_(std::move(lm.lm_)), vm_(std::move(lm.vm_)), index_(lm.index_), weight_(lm.weight_)
{}

float LM::Score(const KenlmState& in, lm::WordIndex index, KenlmState& out) const {
  lm::ngram::State lout;
  float cost = lm_->FullScore(*in, index, lout).prob;
  *out = lout;
  return cost;
}

void LM::BeginSentenceState(KenlmState &b) const {
  *b = lm_->BeginSentenceState();
}

WordPairs::const_iterator LM::begin() const {
  return vm_.begin();    
}

WordPairs::const_iterator LM::end() const {
  return vm_.end();    
}

size_t LM::GetIndex () const {
    return index_;
}

float LM::GetWeight() const {
  return weight_;
}

size_t LM::size() const {
  return vm_.size();
}
