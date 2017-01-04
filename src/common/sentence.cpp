#include <algorithm>
#include "sentence.h"
#include "god.h"
#include "utils.h"
#include "common/vocab.h"

Sentence::Sentence(size_t lineNo, const std::string& line)
: lineNo_(lineNo), line_(line)
{
  std::vector<std::string> tabs;
  Split(line, tabs, "\t");
  size_t i = 0;
  for(auto&& tab : tabs) {
    std::vector<std::string> lineTokens;
    Trim(tab);
    Split(tab, lineTokens, " ");
    auto processed = God::Preprocess(i, lineTokens);
    words_.push_back(God::GetSourceVocab(i++)(processed));
  }
}

const Words& Sentence::GetWords(size_t index) const {
  return words_[index];
}

size_t Sentence::GetLineNum() const {
  return lineNo_;
}

/////////////////////////////////////////////////////////
 Sentences::Sentences()
   : maxLength_(0)
 {
 }

 Sentences::~Sentences()
 {
 }

 void Sentences::push_back(boost::shared_ptr<const Sentence> sentence) {
   const Words &words = sentence->GetWords(0);
   size_t len = words.size();
   if (len > maxLength_) {
     maxLength_ = len;
   }

   coll_.push_back(sentence);
 }

 class LengthOrderer
 {
 public:
   bool operator()(const boost::shared_ptr<const Sentence> &a, const boost::shared_ptr<const Sentence> &b) const
   {
     return a->GetWords(0).size() < b->GetWords(0).size();
   }

 };

 void Sentences::SortByLength()
 {
   std::sort(coll_.begin(), coll_.end(), LengthOrderer());
 }

