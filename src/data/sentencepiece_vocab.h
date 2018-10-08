#pragma once

#ifdef USE_SENTENCEPIECE

#include "data/base_vocab.h"

#include "sentencepiece/src/sentencepiece_processor.h"

#include "3rd_party/exception.h"
#include "3rd_party/yaml-cpp/yaml.h"
#include "common/logging.h"
#include "common/regex.h"
#include "common/utils.h"
#include "common/filesystem.h"

#include <algorithm>
#include <iostream>

namespace marian {

class SentencePieceVocab : public BaseVocab {
private:
  UPtr<sentencepiece::SentencePieceProcessor> spm_;
  float alpha_{0};

public:
  virtual int loadOrCreate(const std::string& vocabPath,
                           const std::string& textPath,
                           int max = 0) override;

  virtual int load(const std::string& vocabPath, int max = 0) override;
  
  virtual Word operator[](const std::string& word) const override;

  virtual Words operator()(const std::vector<std::string>& lineTokens,
                          bool addEOS = true) const override;

  virtual std::vector<std::string> operator()(const Words& sentence,
                                              bool ignoreEOS = true) const;

  virtual const std::string& operator[](Word id) const override;

  virtual Words encode(const std::string& line,
                       bool addEOS = true,
                       bool inference = false) const;

  virtual std::string decode(const Words& sentence,
                             bool ignoreEOS = true) const;

  virtual size_t size() const;

  virtual Word getEosId() const override { return spm_->eos_id(); }
  virtual Word getUnkId() const override { return spm_->unk_id(); }

  void create(const std::string& /*vocabPath*/, const std::string& /*trainPath*/) {
    ABORT("[data] Creating of SentencePieceVocabulary not supported yet");
  }

  void create(io::InputFileStream& /*trainStrm*/,
              io::OutputFileStream& /*vocabStrm*/,
              size_t /*maxSize*/) {
    ABORT("[data] Creating of SentencePieceVocabulary not supported yet");
  }

  void createFake() {
    ABORT("[data] Fake SentencePieceVocabulary not supported");
  }
};

Word SentencePieceVocab::operator[](const std::string& token) const {
  return (Word)spm_->PieceToId(token);
}

Words SentencePieceVocab::operator()(const std::vector<std::string>& lineTokens,
                                     bool addEOS) const {
  Words words(lineTokens.size());
  std::transform(lineTokens.begin(),
                 lineTokens.end(),
                 words.begin(),
                 [&](const std::string& w) { return (*this)[w]; });
  if(addEOS)
    words.push_back(getEosId());
  return words;
}

std::vector<std::string> SentencePieceVocab::operator()(const Words& sentence, bool ignoreEOS) const {
  std::vector<std::string> decoded;
  for(size_t i = 0; i < sentence.size(); ++i) {
    if((sentence[i] != getEosId() || !ignoreEOS)) {
      decoded.push_back((*this)[sentence[i]]);
    }
  }
  return decoded;
}

Words SentencePieceVocab::encode(const std::string& line, bool addEOS, bool inference) const {
  std::vector<std::string> lineTokens;
  if(inference || alpha_ == 0)
    spm_->Encode(line, &lineTokens);
  else
    spm_->SampleEncode(line, -1, alpha_, &lineTokens);
  // @TODO: use numerical sentence directly instead of this call
  return (*this)(lineTokens, addEOS);
}

std::string SentencePieceVocab::decode(const Words& sentence, bool ignoreEOS) const {
  std::string line;
  // @TODO: use numerical sentence directly instead of this call
  std::vector<std::string> lineTokens = (*this)(sentence, ignoreEOS);
  spm_->Decode(lineTokens, &line);
  return line;
}

const std::string& SentencePieceVocab::operator[](Word id) const {
  ABORT_IF(id >= size(), "Unknown word id: ", id);
  return spm_->IdToPiece(id);
}

size_t SentencePieceVocab::size() const {
  return spm_->GetPieceSize();
}

int SentencePieceVocab::loadOrCreate(const std::string& vocabPath,
                                     const std::string& trainPath,
                                     int max) {
  if(vocabPath.empty()) {
    if(filesystem::exists(trainPath + ".spm")) {
      return load(trainPath + ".spm", max);
    }

    // @TODO: make this work, currently it will abort on purpose
    create(trainPath + ".spm", trainPath);
    return load(trainPath + ".spm", max);
  } else {
    if(!filesystem::exists(vocabPath))
      // @TODO: make this work, currently it will abort on purpose
      create(vocabPath, trainPath);
    return load(vocabPath, max);
  }
}

int SentencePieceVocab::load(const std::string& vocabPath, int /*max*/) {
  LOG(info, "[data] Loading SentencePiece vocabulary from file {}", vocabPath);

  ABORT_IF(!filesystem::exists(vocabPath),
           "SentencePiece vocabulary file {} does not exits",
           vocabPath);
  
  spm_.reset(new sentencepiece::SentencePieceProcessor());
  const auto status = spm_->Load(vocabPath);
  
  ABORT_IF(!status.ok(), 
           "SentencePiece error: {}",
           status.ToString());

  return spm_->GetPieceSize();
}

}

#endif
