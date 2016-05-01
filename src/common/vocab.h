#pragma once

#include <map>
#include <string>
#include <vector>
#include <sstream>

#include "types.h"
#include "utils.h"
#include "file_stream.h"

class Vocab {
  public:
    Vocab(const std::string& path) {
        InputFileStream in(path);
        size_t c = 0;
        std::string line;
        while(std::getline((std::istream&)in, line)) {
            str2id_[line] = c++;
            id2str_.push_back(line);
        }
    }

    size_t operator[](const std::string& word) const {
        auto it = str2id_.find(word);
        if(it != str2id_.end())
            return it->second;
        else
            return 1;
    }

    Words operator()(const std::vector<std::string>& lineTokens, bool addEOS = true) const {
      Words words(lineTokens.size());
      std::transform(lineTokens.begin(), lineTokens.end(), words.begin(),
                     [&](const std::string& w) { return (*this)[w]; });
      if(addEOS)
        words.push_back(EOS);
      return words;
    }
    
    Words operator()(const std::string& line, bool addEOS = true) const {
      std::vector<std::string> lineTokens;
      Split(line, lineTokens, " ");
      return (*this)(lineTokens, addEOS);
    }

    std::string operator()(const Words& sentence, bool ignoreEOS = true) const {
      std::stringstream line;
      for(size_t i = 0; i < sentence.size(); ++i) {
        if(sentence[i] != EOS || !ignoreEOS) {
          if(i > 0) {
            line << " ";
          }
          line << (*this)[sentence[i]];
        }
      }
      return line.str();
    }


    const std::string& operator[](size_t id) const {
        return id2str_[id];
    }

    size_t size() const {
      return id2str_.size();
    }

  private:
    std::map<std::string, size_t> str2id_;
    std::vector<std::string> id2str_;
};
