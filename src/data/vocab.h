#pragma once

#include "data/vocab_impl.h"

namespace marian {

// Wrapper around vocabulary types. Can choose underlying
// vocabulary implementation (vImpl_) based on speficied path
// and suffix.
// Vocabulary implementations can currently be:
// * DefaultVocabulary for YAML (*.yml and *.yaml) and TXT (any other non-specific ending)
// * SentencePiece with suffix *.spm (works, but has to be created outside Marian)
class Vocab {
private:
  Ptr<VocabImpl> vImpl_;

public:
  virtual int loadOrCreate(const std::string& vocabPath,
                           const std::string& textPath,
                           int max = 0);

  virtual int load(const std::string& vocabPath, int max = 0);
  virtual void create(const std::string& vocabPath, const std::string& trainPath);

  virtual void create(io::InputFileStream& trainStrm,
                      io::OutputFileStream& vocabStrm,
                      size_t maxSize = 0);

  // string token to token id
  virtual Word operator[](const std::string& word) const {
    return vImpl_->operator[](word);
  }

  // token id to string token
  virtual const std::string& operator[](Word id) const {
    return vImpl_->operator[](id);
  }

  // line of text to list of token ids, can perform tokenization
  virtual Words encode(const std::string& line,
                       bool addEOS = true,
                       bool inference = false) const {
    return vImpl_->encode(line, addEOS, inference);
  }

  // list of token ids to single line, can perform detokenization
  virtual std::string decode(const Words& sentence,
                             bool ignoreEOS = true) const {
    return vImpl_->decode(sentence, ignoreEOS);
  }

  // number of vocabulary items
  virtual size_t size() const { return vImpl_->size(); }

  // return EOS symbol id
  virtual Word getEosId() const { return vImpl_->getEosId(); }

  // return UNK symbol id
  virtual Word getUnkId() const { return vImpl_->getUnkId(); }

  // create fake vocabulary for collecting batch statistics
  virtual void createFake();
};

}  // namespace marian
