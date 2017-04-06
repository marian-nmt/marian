#pragma once

#include <string>
#include <memory>
#include <unordered_set>
#include <set>

#include "common/types.h"

namespace amunmt {

class Vocab;

class Filter {
  public:
    Filter(const size_t numFirstWords=10000);

    Filter(const Vocab& srcVocab,
           const Vocab& trgVocab,
           const std::string& path,
           const size_t numFirstWords=10000,
           const size_t maxNumTranslation=1000);

    template<class T>
    Words GetFilteredVocab(const T& srcWords, const size_t maxVocabSize) const {
      std::set<Word> filtered;

      for(size_t i = 0; i < std::min(numFirstWords_, maxVocabSize); ++i) {
        filtered.insert(i);
      }

      for (const auto& srcWord : srcWords) {
        for (const auto& trgWord : mapper_[srcWord]) {
          if (trgWord < maxVocabSize) {
            filtered.insert(trgWord);
          }
        }
      }

      Words output(filtered.begin(), filtered.end());

      return output;
    }

    size_t GetNumFirstWords() const;

    void SetNumFirstWords(size_t numFirstWords);

    static std::vector<Words> ParseAlignmentFile(const Vocab& srcVocab,
                                                 const Vocab& trgVocab,
                                                 const std::string& path,
                                                 const size_t maxNumTranslation,
                                                 const size_t numNFirst);

  private:
    size_t numFirstWords_;
    const std::vector<Words> mapper_;
};

typedef std::unique_ptr<Filter> FilterPtr;

}

