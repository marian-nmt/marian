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

    SentencePtr at(unsigned id) const;
    const Sentence &Get(unsigned id) const;

    unsigned size() const;

    unsigned GetMaxLength() const;

    void SortByLength();

    SentencesPtr NextMiniBatch(unsigned batchsize, int batchWords);

    std::string Debug(unsigned verbosity = 1) const;

  protected:
    std::vector<SentencePtr> coll_;
    unsigned maxLength_;

    Sentences(const Sentences &) = delete;
};

}

