#pragma once

#include <map>
#include <string>
#include <vector>

#include "common/types.h"

namespace amunmt {

class Vocab {
  public:
    Vocab(const std::string& path);

    size_t operator[](const std::string& word) const;

    Words operator()(const std::vector<std::string>& lineTokens, bool addEOS = true) const;

    Words operator()(const std::string& line, bool addEOS = true) const;

    std::vector<std::string> operator()(const Words& sentence, bool ignoreEOS = true) const;

    const std::string& operator[](size_t id) const;

    size_t size() const;

  private:
    std::map<std::string, size_t> str2id_;
    std::vector<std::string> id2str_;
};

}
