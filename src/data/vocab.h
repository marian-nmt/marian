#pragma once

#include "data/base_vocab.h"

namespace marian {

// Wrapper around vocabulary types. Can choose underlying
// vocabulary implementation (vImpl_) based on speficied path
// and suffix.
// Vocabulary implementations can currently be:
// * DefaultVocabulary (YAML and TXT)
// * SentencePiece (in preparation)
class Vocab : public BaseVocab {
private:
  Ptr<BaseVocab> vImpl_;

public:
  virtual int loadOrCreate(const std::string& vocabPath,
                           const std::string& textPath,
                           int max = 0) override;

  virtual int load(const std::string& vocabPath, int max = 0) override;
  virtual void create(const std::string& vocabPath, const std::string& trainPath) override;
  virtual void create(io::InputFileStream& trainStrm,
                      io::OutputFileStream& vocabStrm,
                      size_t maxSize = 0) override;

  // string token to token id
  virtual Word operator[](const std::string& word) const override {
    return vImpl_->operator[](word);
  }

  // tokenized text to list of token ids
  virtual Words operator()(const std::vector<std::string>& lineTokens,
                          bool addEOS = true) const override {
    return vImpl_->operator()(lineTokens, addEOS);
  }

  // list of token ids to tokenized text
  virtual std::vector<std::string> operator()(const Words& sentence,
                                              bool ignoreEOS = true) const override {
    return vImpl_->operator()(sentence, ignoreEOS);
  }

  // token id to string token
  virtual const std::string& operator[](Word id) const override {
    return vImpl_->operator[](id);
  }

  // line of text to list of token ids, can perform tokenization
  virtual Words encode(const std::string& line,
                       bool addEOS = true,
                       bool inference = false) const override {
    return vImpl_->encode(line, addEOS, inference);
  }

  // list of token ids to single line, can perform detokenization
  virtual std::string decode(const Words& sentence,
                             bool ignoreEOS = true) const override {
    return vImpl_->decode(sentence, ignoreEOS);
  }

  // number of vocabulary items
  virtual size_t size() const override { return vImpl_->size(); }

  // return EOS symbol id
  virtual Word getEosId() const override { return vImpl_->getEosId(); }

  // return UNK symbol id
  virtual Word getUnkId() const override { return vImpl_->getUnkId(); }

  // create fake vocabulary for collecting batch statistics
  virtual void createFake() override;

  // change underlying preprocessor: deprecated and to be removed.
  // will be replaced with specific vocabulary implementation with
  // specific built-in processor
  virtual void resetProcessor(Ptr<Processor> processor) override {
    vImpl_->resetProcessor(processor);
  };
};

}  // namespace marian
