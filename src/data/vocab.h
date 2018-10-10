#pragma once

#include "common/file_stream.h"
#include "data/types.h"

#include <map>
#include <string>
#include <vector>

namespace marian {

class Vocab {
public:
  Vocab();

  size_t operator[](const std::string& word) const;

  Words operator()(const std::vector<std::string>& lineTokens,
                   bool addEOS = true) const;

  Words operator()(const std::string& line, bool addEOS = true) const;

  std::vector<std::string> operator()(const Words& sentence,
                                      bool ignoreEOS = true) const;

  const std::string& operator[](Word id) const;

  size_t size() const;

  int loadOrCreate(const std::string& vocabPath,
                   const std::string& textPath,
                   int max = 0);
  int load(const std::string& vocabPath, int max = 0);
  void create(const std::string& vocabPath, const std::string& trainPath);
  void create(io::InputFileStream& trainStrm,
              io::OutputFileStream& vocabStrm,
              size_t maxSize = 0);

  Word getEosId() const { return eosId_; }
  Word getUnkId() const { return unkId_; }

  void createFake();  // for fakeBatch()

private:
  Word insertWord(Word id, const std::string& str);

private:
  typedef std::map<std::string, Word> Str2Id;
  Str2Id str2id_;

  typedef std::vector<std::string> Id2Str;
  Id2Str id2str_;

  Word eosId_ = (Word)-1;
  Word unkId_ = (Word)-1;

  class VocabFreqOrderer;
};
}  // namespace marian
