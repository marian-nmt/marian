#pragma once

#include <map>
#include <string>
#include <vector>
#include <sstream>
#include <yaml-cpp/yaml.h>

#include "types.h"
#include "utils.h"
#include "file_stream.h"
#include "exception.h"

class Vocab {
  public:
    Vocab(const std::string& path) {
        YAML::Node vocab = YAML::Load(InputFileStream(path));
        for(auto&& pair : vocab) {
          auto str = pair.first.as<std::string>();
          auto id = pair.second.as<Word>();
          str2id_[str] = id;
          if(id >= id2str_.size())
            id2str_.resize(id + 1);
          id2str_[id] = str;
        }
        UTIL_THROW_IF2(id2str_.empty(), "Empty vocabulary " << path);
        id2str_[0] = "</s>";
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

    std::vector<std::string> operator()(const Words& sentence, bool ignoreEOS = true) const {
      std::vector<std::string> decoded;
      for(size_t i = 0; i < sentence.size(); ++i) {
        if(sentence[i] != EOS || !ignoreEOS) {
          decoded.push_back((*this)[sentence[i]]);
        }
      }
      return decoded;
    }


    const std::string& operator[](size_t id) const {
      UTIL_THROW_IF2(id >= id2str_.size(), "Unknown word id: " << id);
      return id2str_[id];
    }

    size_t size() const {
      return id2str_.size();
    }

  private:
    std::map<std::string, size_t> str2id_;
    std::vector<std::string> id2str_;
};
