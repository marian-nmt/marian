#pragma once

#include <string>
#include <memory>

#include "common/types.h"

class Filter {
  public:
    Filter(const size_t numFirstWords=10000);

    Filter(const std::string& path, const size_t numFirstWords=10000);

    Words GetFilteredVocab(const Words& srcWords, const size_t maxVocabSize) const;

    size_t GetNumFirstWords() const;

    void SetNumFirstWords(size_t numFirstWords);

    static std::vector<Words> ParseAlignmentFile(const std::string& path);

  private:
    size_t numFirstWords_;
    const std::vector<Words> mapper_;
};

typedef std::unique_ptr<Filter> FilterPtr;
