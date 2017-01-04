#pragma once
#include <vector>
#include <string>
#include <boost/shared_ptr.hpp>
#include "types.h"

class Sentence {
  public:
    Sentence(size_t lineNo, const std::string& line);

    const Words& GetWords(size_t index = 0) const;

    size_t GetLineNum() const;

  private:
    std::vector<Words> words_;
    size_t lineNo_;
    std::string line_;
};


/////////////////////////////////////////////////////////
 class Sentences
 {
 public:
   Sentences();
   ~Sentences();

   void push_back(const Sentence *sentence);

   boost::shared_ptr<const Sentence> at(size_t id) const {
     return coll_.at(id);
   }

   size_t size() const {
     return coll_.size();
   }

   size_t GetMaxLength() const {
     return maxLength_;
   }

 protected:
   typedef  std::vector< boost::shared_ptr<const Sentence> > Coll;
   Coll coll_;

   size_t maxLength_;
 };
