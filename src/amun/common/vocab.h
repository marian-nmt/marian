#pragma once

#include <map>
#include <string>
#include <vector>

#include "common/types.h"

namespace amunmt {

class Vocab {
  public:
    Vocab(const std::string& path);

    unsigned operator[](const std::string& word) const;

    Words operator()(const std::vector<std::string>& lineTokens, bool addEOS = true) const;

    Words operator()(const std::string& line, bool addEOS = true) const;

    std::vector<std::string> operator()(const Words& sentence, bool ignoreEOS = true) const;

    const std::string& operator[](unsigned id) const;

    unsigned size() const;

  private:
    typedef std::map<std::string, unsigned> Str2Id;
    Str2Id str2id_;

    typedef std::vector<std::string> Id2Str;
    Id2Str id2str_;

};

}
