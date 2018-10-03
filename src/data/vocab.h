#pragma once

#include "common/definitions.h"
#include "common/file_stream.h"
#include "common/options.h"
#include "data/types.h"
#include "common/utils.h"
#include "sentencepiece/src/sentencepiece_processor.h"

#include <map>
#include <string>
#include <vector>

namespace marian {

class Processor {
public:
  virtual void encode(const std::string& line, std::vector<std::string>& pieces) const {
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
    spm_->Load(spmModel);
  }

   void encode(const std::string& line, std::vector<std::string>& pieces) const override {
    if(alpha_ != 0)
      spm_->SampleEncode(line, -1, alpha_, &pieces);
    else
      spm_->Encode(line, &pieces);
  }

   void decode(const std::vector<std::string>& pieces, std::string& line) const override {
    spm_->Decode(pieces, &line);
  }
};

class Vocab {
public:
  Vocab();

  int loadOrCreate(const std::string& vocabPath,
                   const std::string& textPath,
                   int max = 0);

  int load(const std::string& vocabPath, int max = 0);
  void create(const std::string& vocabPath, const std::string& trainPath);
  void create(io::InputFileStream& trainStrm,
              io::OutputFileStream& vocabStrm,
              size_t maxSize = 0);

   void resetProcessor(Ptr<Processor> processor) {
     processor_ = processor;
   }

  size_t operator[](const std::string& word) const;

  Words operator()(const std::vector<std::string>& lineTokens,
                          bool addEOS = true) const;

  Words operator()(const std::string& line, bool addEOS = true) const;

  std::vector<std::string> operator()(const Words& sentence,
                                      bool ignoreEOS = true) const;

  std::string decode(const Words& sentence,
                     bool reverse = false) const;

  const std::string& operator[](size_t id) const;

  size_t size() const;

  Word GetEosId() const { return eosId_; }
  Word GetUnkId() const { return unkId_; }

  void createFake();  // for fakeBatch()

private:
  Word insertWord(Word id, const std::string& str);

private:
  typedef std::map<std::string, size_t> Str2Id;
  Str2Id str2id_;

  typedef std::vector<std::string> Id2Str;
  Id2Str id2str_;

  Word eosId_ = (Word)-1;
  Word unkId_ = (Word)-1;

  class VocabFreqOrderer;

  Ptr<Processor> processor_;
};

}  // namespace marian
