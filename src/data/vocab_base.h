#pragma once

#include "data/types.h"
#include "common/definitions.h"
#include "common/utils.h"
#include "common/file_stream.h"

namespace marian {

class IVocab {
public:
  virtual size_t load(const std::string& vocabPath, size_t maxSize = 0) = 0;

  virtual void create(const std::string& vocabPath,
                      const std::vector<std::string>& trainPaths,
                      size_t maxSize) = 0;

  // return canonical suffix for given type of vocabulary
  virtual const std::string& canonicalExtension() const = 0;
  virtual const std::vector<std::string>& suffixes() const = 0;

  size_t findAndLoad(const std::string& path, size_t maxSize) { // @TODO: Only used in one place; just inline it there -> true interface
    for(auto suffix : suffixes())
      if(filesystem::exists(path + suffix))
        return load(path + suffix, maxSize);
    return 0;
  }

  virtual Word operator[](const std::string& word) const = 0;

  virtual Words encode(const std::string& line,
                       bool addEOS = true,
                       bool inference = false) const = 0;

  virtual std::string decode(const Words& sentence,
                             bool ignoreEos = true) const = 0;
  virtual std::string surfaceForm(const Words& sentence) const = 0;

  virtual const std::string& operator[](Word id) const = 0;

  virtual size_t size() const = 0;
  virtual size_t lemmaSize() const { return size(); }
  virtual std::string type() const = 0;

  virtual Word getEosId() const = 0;
  virtual Word getUnkId() const = 0;

  // without specific knowledge of tokenization, these two functions can do nothing
  // Both SentencePieceVocab and FactoredSegmenterVocab
  virtual std::string toUpper(const std::string& line) const { return line; }
  virtual std::string toEnglishTitleCase(const std::string& line) const { return line; }

  // Identity mapping for default vocabularies, hence do nothing
  virtual void transcodeToShortlistInPlace(WordIndex* ptr, size_t num) const { ptr; num; }

  // Populates vector `special` with special words like "\n" etc.
  virtual void addSpecialWords(std::vector<Word>& special) const { special; }

  virtual void createFake() = 0;

  virtual Word randWord() const {
    return Word::fromWordIndex(rand() % size());
  }
  virtual ~IVocab() {};
};

class Options;
Ptr<IVocab> createDefaultVocab();
Ptr<IVocab> createClassVocab();
Ptr<IVocab> createSentencePieceVocab(const std::string& vocabPath, Ptr<Options>, size_t batchIndex);
Ptr<IVocab> createFactoredVocab(const std::string& vocabPath);

}
