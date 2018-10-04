#pragma once

#include "data/processor.h"
#include "common/definitions.h"
#include "common/file_stream.h"
#include "common/options.h"
#include "data/types.h"

#include <map>
#include <string>
#include <vector>

namespace marian {

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

  std::vector<std::string> operator()(const Words& sentence,
                                      bool ignoreEOS = true) const;

  Words encode(const std::string& line, 
               bool addEOS = true,
               bool inference = false) const;

  std::string decode(const Words& sentence,
                     bool ignoreEos = true) const;

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
