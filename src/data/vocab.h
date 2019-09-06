#pragma once

#include "common/definitions.h"
#include "data/types.h"
#include "common/options.h"
#include "common/file_stream.h"

namespace marian {

class IVocab;

// Wrapper around vocabulary types. Can choose underlying
// vocabulary implementation (vImpl_) based on speficied path
// and suffix.
// Vocabulary implementations can currently be:
// * DefaultVocabulary for YAML (*.yml and *.yaml) and TXT (any other non-specific ending)
// * SentencePiece with suffix *.spm (works, but has to be created outside Marian)
class Vocab {
private:
  Ptr<IVocab> vImpl_;
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

  // token index to string token
  const std::string& operator[](Word word) const;

  // line of text to list of token ids, can perform tokenization
  Words encode(const std::string& line,
               bool addEOS = true,
               bool inference = false) const;

  // convert sequence of token ids to single line, can perform detokenization
  std::string decode(const Words& sentence,
                     bool ignoreEOS = true) const;

  // convert sequence of token its to surface form (incl. removng spaces, applying factors)
  // for in-process BLEU validation
  std::string surfaceForm(const Words& sentence) const;

  // number of vocabulary items
  size_t size() const;

  // number of vocabulary items
  std::string type() const;

  // return EOS symbol id
  Word getEosId() const;

  // return UNK symbol id
  Word getUnkId() const;

  // for corpus augmentation: convert string to all-caps
  // @TODO: Consider a different implementation where this does not show on the vocab interface,
  //        but instead as additional options passed to vocab instantiation.
  std::string toUpper(const std::string& line) const;

  // for corpus augmentation: convert string to title case
  std::string toEnglishTitleCase(const std::string& line) const;

  // for short-list generation
  void transcodeToShortlistInPlace(WordIndex* ptr, size_t num) const;

  // create fake vocabulary for collecting batch statistics
  void createFake();

  // generate a fake word (using rand())
  Word randWord();

  // give access to base implementation. Returns null if not the requested type.
  template<class VocabType> // e.g. FactoredVocab
  Ptr<VocabType> tryAs() const { return std::dynamic_pointer_cast<VocabType>(vImpl_); }
};

}  // namespace marian
