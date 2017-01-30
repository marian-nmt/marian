#pragma once
#include <memory>
#include <vector>
#include <string>
#include "types.h"

namespace amunmt {

class God;

class Sentence {
  public:

    Sentence(const God &god, size_t vLineNum, const std::string& line);
    Sentence(const God &god, size_t vLineNum, const std::vector<std::string>& words);
		Sentence(God &god, size_t lineNum, const std::vector<size_t>& words);

    const Words& GetWords(size_t index = 0) const;
    size_t GetLineNum() const;

  private:
    std::vector<Words> words_;
    size_t lineNum_;

    Sentence(const Sentence &) = delete;
};

using SentencePtr = std::shared_ptr<Sentence>;


class Sentences {
 public:
  Sentences(size_t taskCounter = 0, size_t bunchId = 0);
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

  // for debugging only. Do not use to assign to thread, GPU etc
  size_t GetTaskCounter() const {
    return taskCounter_;
  }

  // for debugging only. Do not use to assign to thread, GPU etc
  size_t GetBunchId() const {
    return bunchId_;
  }

  void SortByLength();

 protected:
   std::vector<SentencePtr> coll_;
   size_t taskCounter_;
   size_t bunchId_;
   size_t maxLength_;

   Sentences(const Sentences &) = delete;
};

}

