#pragma once

#include "sentence.h"

namespace amunmt {

class Sentences;
using SentencesPtr = std::shared_ptr<Sentences>;

class Sentences {
  public:
    Sentences();
    Sentences(const Sentences &other);
    ~Sentences();

    void push_back(SentencePtr sentence);

    SentencePtr at(size_t id) const;
    const Sentence &Get(size_t id) const;

    void Set(size_t id, SentencePtr sentence);

    size_t size() const;

    size_t GetMaxLength() const;

    void RecalcMaxLength();

    void SortByLength();

    SentencesPtr NextMiniBatch(size_t batchsize, int batchWords);

  protected:
    std::vector<SentencePtr> coll_;
    size_t maxLength_;
};

}

