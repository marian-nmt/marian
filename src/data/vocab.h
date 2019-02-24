#pragma once

#include "common/definitions.h"
#include "data/types.h"
#include "common/options.h"
#include "common/file_stream.h"

namespace marian {

class VocabBase;

// Wrapper around vocabulary types. Can choose underlying
// vocabulary implementation (vImpl_) based on speficied path
// and suffix.
// Vocabulary implementations can currently be:
// * DefaultVocabulary for YAML (*.yml and *.yaml) and TXT (any other non-specific ending)
// * SentencePiece with suffix *.spm (works, but has to be created outside Marian)
class Vocab {
private:
  Ptr<VocabBase> vImpl_;
  Ptr<Options> options_;
  size_t batchIndex_;

public:
  Vocab(Ptr<Options> options, size_t batchIndex)
  : options_(options), batchIndex_(batchIndex) {}

  size_t loadOrCreate(const std::string& vocabPath,
                      const std::vector<std::string>& trainPaths,
                      size_t maxSize = 0);

  size_t load(const std::string& vocabPath, size_t maxSize = 0);

  void create(const std::string& vocabPath,
              const std::vector<std::string>& trainPaths,
              size_t maxSize);

  void create(const std::string& vocabPath,
              const std::string& trainPath,
              size_t maxSize);

  // string token to token id
  Word operator[](const std::string& word) const;

  // token id to string token
  const std::string& operator[](Word id) const;

  // line of text to list of token ids, can perform tokenization
  Words encode(const std::string& line,
               bool addEOS = true,
               bool inference = false) const;

  // list of token ids to single line, can perform detokenization
  std::string decode(const Words& sentence,
                     bool ignoreEOS = true) const;

  // number of vocabulary items
  size_t size() const;

  // number of vocabulary items
  std::string type() const;

  // return EOS symbol id
  Word getEosId() const;

  // return UNK symbol id
  Word getUnkId() const;

  // create fake vocabulary for collecting batch statistics
  void createFake();
};

}  // namespace marian
