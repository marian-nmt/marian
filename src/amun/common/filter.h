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
    Filter(const unsigned numFirstWords=10000);

    Filter(const Vocab& srcVocab,
           const Vocab& trgVocab,
           const std::string& path,
           const unsigned numFirstWords=10000,
           const unsigned maxNumTranslation=1000);

    template<class T>
    Words GetFilteredVocab(const T& srcWords, const unsigned maxVocabSize) const {
      std::set<Word> filtered;

      for(unsigned i = 0; i < std::min(numFirstWords_, maxVocabSize); ++i) {
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

    unsigned GetNumFirstWords() const;

    void SetNumFirstWords(unsigned numFirstWords);

    static std::vector<Words> ParseAlignmentFile(const Vocab& srcVocab,
                                                 const Vocab& trgVocab,
                                                 const std::string& path,
                                                 const unsigned maxNumTranslation,
                                                 const unsigned numNFirst);

  private:
    unsigned numFirstWords_;
    const std::vector<Words> mapper_;
};

typedef std::unique_ptr<Filter> FilterPtr;

}

