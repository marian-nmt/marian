#pragma once
#include "sentence.h"

namespace amunmt {

class God;
class ThreadPool;
class Sentences;
using SentencesPtr = std::shared_ptr<Sentences>;

class Sentences {
 public:
  Sentences();
  ~Sentences();

  void push_back(SentencePtr sentence);

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

 protected:
   std::vector<SentencePtr> coll_;
   size_t maxLength_;

   Sentences(const Sentences &) = delete;
};

}

