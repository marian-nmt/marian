#pragma once

#include "common/definitions.h"
#include "common/utils.h"
#include "sentencepiece/src/sentencepiece_processor.h"

#include <map>
#include <string>
#include <vector>

namespace marian {

class Processor {
public:
  virtual void encode(const std::string& line, std::vector<std::string>& pieces, bool /*inference*/) const {
    utils::split(line, pieces, " ");
  }

  virtual void decode(const std::vector<std::string>& pieces, std::string& line) const {
    line = utils::join(pieces, " ");
  }
};

class SentencePiece : public Processor {
private:
   UPtr<sentencepiece::SentencePieceProcessor> spm_;
   float alpha_{0};

public:
  SentencePiece(const std::string& spmModel, float alpha = 0) 
    : spm_(new sentencepiece::SentencePieceProcessor()), alpha_(alpha) {
    LOG(info, "Loading SentencePiece model from {} with alpha {}", spmModel, alpha);
    spm_->Load(spmModel);
  }

  void encode(const std::string& line, std::vector<std::string>& pieces, bool inference = false) const override {
    if(!inference && alpha_ != 0)
      spm_->SampleEncode(line, -1, alpha_, &pieces);
    else
      spm_->Encode(line, &pieces);
  }

  void decode(const std::vector<std::string>& pieces, std::string& line) const override {
    spm_->Decode(pieces, &line);
  }
};

}