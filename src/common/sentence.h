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

