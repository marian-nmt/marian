#pragma once

#ifdef USE_SENTENCEPIECE

#include "data/vocab_impl.h"

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

class SentencePieceVocab : public VocabImpl {
private:
  UPtr<sentencepiece::SentencePieceProcessor> spm_;
  float alpha_{0};
  std::vector<std::string> suffixes_ = {".spm"};

  Ptr<Options> options_;
  size_t batchIndex_{0};

public:
  SentencePieceVocab(Ptr<Options> options, size_t batchIndex)
    : options_(options), batchIndex_(batchIndex) {

    if(options_->has("sentencepiece-alphas")) {
      auto alphas = options_->get<std::vector<float>>("sentencepiece-alphas");
      if(alphas.size() <= batchIndex)
        alpha_ = 0.f;
      else
        alpha_ = alphas[batchIndex_];

      if(alpha_ > 0)
        LOG(debug,
            "Setting SentencePieceVocab sampling factor to {} for input {}",
            alpha_,
            batchIndex_);
    }

  }

  virtual int load(const std::string& vocabPath, int max = 0) override;

  virtual const std::string& canonicalSuffix() const { return suffixes_[0]; }
  virtual const std::vector<std::string>& suffixes() const { return suffixes_; }

  virtual std::string suffix() { return suffixes_[0]; };

  virtual Word operator[](const std::string& word) const override;

  virtual const std::string& operator[](Word id) const override;

  virtual Words encode(const std::string& line,
                       bool addEOS = true,
                       bool inference = false) const;

  virtual std::string decode(const Words& sentence,
                             bool ignoreEOS = true) const;

  virtual size_t size() const;

  virtual std::string type() const { return "SentencePieceVocab"; }

  virtual Word getEosId() const override { return (Word)spm_->eos_id(); }
  virtual Word getUnkId() const override { return (Word)spm_->unk_id(); }

  void create(const std::string& /*vocabPath*/, const std::string& /*trainPath*/) {
    ABORT("[data] Training of SentencePieceVocab not yet supported");
  }

  void create(io::InputFileStream& /*trainStrm*/,
              io::OutputFileStream& /*vocabStrm*/,
              size_t /*maxSize*/) {
    ABORT("[data] Training of SentencePieceVocab not yet supported");
  }

  void createFake() {
    ABORT("[data] Fake SentencePieceVocab not supported");
  }
};

Word SentencePieceVocab::operator[](const std::string& token) const {
  return (Word)spm_->PieceToId(token);
}

const std::string& SentencePieceVocab::operator[](Word id) const {
  ABORT_IF(id >= size(), "Unknown word id: ", id);
  return spm_->IdToPiece(id);
}

Words SentencePieceVocab::encode(const std::string& line, bool addEOS, bool inference) const {
  std::vector<int> spmIds;
  if(inference || alpha_ == 0)
    spm_->Encode(line, &spmIds);
  else
    spm_->SampleEncode(line, -1, alpha_, &spmIds);

  Words words(spmIds.begin(), spmIds.end());

  if(addEOS)
    words.push_back(getEosId());
  return words;
}

std::string SentencePieceVocab::decode(const Words& sentence, bool ignoreEOS) const {
  std::string line;
  // convert vector of Word to vector of int
  std::vector<int> spmSentence(sentence.begin(), sentence.end());
  spm_->Decode(spmSentence, &line);
  return line;
}

size_t SentencePieceVocab::size() const {
  return spm_->GetPieceSize();
}

int SentencePieceVocab::load(const std::string& vocabPath, int /*max*/) {
  LOG(info, "[data] Loading SentencePieceVocab from file {}", vocabPath);

  ABORT_IF(!filesystem::exists(vocabPath),
           "SentencePieceVocab file {} does not exits",
           vocabPath);

  spm_.reset(new sentencepiece::SentencePieceProcessor());
  const auto status = spm_->Load(vocabPath);

  ABORT_IF(!status.ok(),
           "SentencePieceVocab error: {}",
           status.ToString());

  return spm_->GetPieceSize();
}

}

#endif
