#pragma once

#include "god.h"

class Sentence {
  public:
    Sentence(size_t lineNo, const std::string& line)
    : lineNo_(lineNo), line_(line), words_(God::GetSourceVocab()(line))
    {}
    
    const Words& GetWords() const {
      return words_;
    }
    
    size_t GetLine() const {
      return lineNo_;
    }
    
  private:
    Words words_;
    size_t lineNo_;
    std::string line_;
};

