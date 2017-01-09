#pragma once
#include <vector>
#include <string>
#include <boost/shared_ptr.hpp>
#include "types.h"

class Sentence {
  public:
    size_t lineNum;

    Sentence(size_t vLineNum, const std::string& line);

    const Words& GetWords(size_t index = 0) const;

  private:
    std::vector<Words> words_;
    std::string line_;

    Sentence(const Sentence &) = delete;
};


/////////////////////////////////////////////////////////
 class Sentences
 {
 public:
  size_t taskCounter;
  size_t bunchId;

   Sentences(size_t vTaskCounter = 0, size_t vBunchId = 0);
   ~Sentences();

   void push_back(boost::shared_ptr<const Sentence> sentence);

   boost::shared_ptr<const Sentence> at(size_t id) const {
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
   typedef  std::vector< boost::shared_ptr<const Sentence> > Coll;
   Coll coll_;

   size_t maxLength_;

   Sentences(const Sentences &) = delete;
};
