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
  int loadOrCreate(const std::string& vocabPath,
                   const std::string& textPath,
                   int max = 0);

  int load(const std::string& vocabPath, int max = 0);
  void create(const std::string& vocabPath, const std::string& trainPath);

  void create(io::InputFileStream& trainStrm,
              io::OutputFileStream& vocabStrm,
              size_t maxSize = 0);

  // string token to token id
  Word operator[](const std::string& word) const {
    return vImpl_->operator[](word);
  }

  // token id to string token
  const std::string& operator[](Word id) const {
    return vImpl_->operator[](id);
  }

  // line of text to list of token ids, can perform tokenization
  Words encode(const std::string& line,
               bool addEOS = true,
               bool inference = false) const {
    return vImpl_->encode(line, addEOS, inference);
  }

  // list of token ids to single line, can perform detokenization
  std::string decode(const Words& sentence,
                     bool ignoreEOS = true) const {
    return vImpl_->decode(sentence, ignoreEOS);
  }

  // number of vocabulary items
  size_t size() const { return vImpl_->size(); }

  // number of vocabulary items
  std::string type() const { return vImpl_->type(); }

  // return EOS symbol id
  Word getEosId() const { return vImpl_->getEosId(); }

  // return UNK symbol id
  Word getUnkId() const { return vImpl_->getUnkId(); }

  // create fake vocabulary for collecting batch statistics
  void createFake();
};

}  // namespace marian
