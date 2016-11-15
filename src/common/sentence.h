#pragma once
#include <vector>
#include <string>
#include "types.h"

class Sentence {
  public:
    Sentence(size_t lineNo, const std::string& line);

    const Words& GetWords(size_t index = 0) const;

    size_t GetLine() const;

  private:
    std::vector<Words> words_;
    size_t lineNo_;
    std::string line_;
};

using Sentences = std::vector<Sentence>;

/////////////////////////////////////////////////////////
// class Sentences
// {
// public:
  // Sentences();

  // void push_back(const Sentence *sentence);

  // const Sentence* at(size_t id) const {
    // return coll_.at(id);
  // }

  // size_t size() const {
    // return coll_.size();
  // }

  // size_t GetMaxLength() const {
    // return maxLength_;
  // }

// protected:
  // typedef  std::vector<const Sentence*> Coll;
  // Coll coll_;

  // size_t maxLength_;
// };
