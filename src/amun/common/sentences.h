#pragma once

#include "sentence.h"

namespace amunmt {

class Sentences;
using SentencesPtr = std::shared_ptr<Sentences>;

class Sentences {
 protected:
   std::vector<SentencePtr> coll_;
   size_t maxLength_;

 public:
  Sentences();
  ~Sentences();

  void push_back(SentencePtr sentence);

  auto begin() const -> decltype(coll_.cbegin()) {
    return coll_.begin();
  }

  auto end() const -> decltype(coll_.cend()) {
    return coll_.begin();
  }

  SentencePtr at(size_t id) const {
    return coll_.at(id);
  }

  size_t size() const {
    return coll_.size();
  }

  size_t GetMaxLength() const {
    return maxLength_;
  }

  void SortByLength();

  SentencesPtr NextMiniBatch(size_t batchsize);


   Sentences(const Sentences &) = delete;
};

}

