#pragma once

#include "sentence.h"

namespace amunmt {

class Sentences;
using SentencesPtr = std::shared_ptr<Sentences>;

class Sentences {
  public:
    Sentences();
    ~Sentences();

    void push_back(SentencePtr sentence);

    SentencePtr at(size_t id) const;

    size_t size() const;

    size_t GetMaxLength() const;

    void SortByLength();

    SentencesPtr NextMiniBatch(size_t batchsize, int batchWords);

  protected:
    std::vector<SentencePtr> coll_;
    size_t maxLength_;

    Sentences(const Sentences &) = delete;
};

}

