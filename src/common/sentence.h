#pragma once
#include <memory>
#include <vector>
#include <string>
#include "types.h"

class God;

class Sentence {
  public:

    Sentence(God &god, size_t lineNum, const std::string& line);
    Sentence(God &god, size_t lineNum, const std::vector<std::string>& words);
    Sentence(God &god, size_t lineNum, const std::vector<size_t>& words);

    const Words& GetWords(size_t index = 0) const;
    size_t GetLineNum() const;

  private:
    std::vector<Words> words_;
    std::string line_;
    size_t lineNum_;

    Sentence(const Sentence &) = delete;
};

using SentencePtr = std::shared_ptr<Sentence>;


class Sentences {
 public:
  size_t taskCounter;
  size_t bunchId;

  Sentences(size_t vTaskCounter = 0, size_t vBunchId = 0);
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

 protected:
   std::vector<SentencePtr> coll_;

   size_t maxLength_;

   Sentences(const Sentences &) = delete;
};
