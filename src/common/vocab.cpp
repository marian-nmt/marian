#include "common/vocab.h"

#include <sstream>
#include <yaml-cpp/yaml.h>

#include "common/utils.h"
#include "common/file_stream.h"
#include "common/exception.h"

namespace amunmt {

Vocab::Vocab(const std::string& path) {
    YAML::Node vocab = YAML::Load(InputFileStream(path));
    for(auto&& pair : vocab) {
      auto str = pair.first.as<std::string>();
      auto id = pair.second.as<Word>();
      str2id_[str] = id;
      if(id >= id2str_.size())
        id2str_.resize(id + 1);
      id2str_[id] = str;
    }
    amunmt_UTIL_THROW_IF2(id2str_.empty(), "Empty vocabulary " << path);
    id2str_[EOS_ID] = EOS_STR;
    id2str_[UNK_ID] = UNK_STR;
}

size_t Vocab::operator[](const std::string& word) const {
    auto it = str2id_.find(word);
    if(it != str2id_.end())
        return it->second;
    else
        return UNK_ID;
}

Words Vocab::operator()(const std::vector<std::string>& lineTokens, bool addEOS) const {
  Words words(lineTokens.size());
  std::transform(lineTokens.begin(), lineTokens.end(), words.begin(),
                  [&](const std::string& w) { return (*this)[w]; });
  if(addEOS)
    words.push_back(EOS_ID);
  return words;
}

Words Vocab::operator()(const std::string& line, bool addEOS) const {
  std::vector<std::string> lineTokens;
  Split(line, lineTokens, " ");
  return (*this)(lineTokens, addEOS);
}

std::vector<std::string> Vocab::operator()(const Words& sentence, bool ignoreEOS) const {
  std::vector<std::string> decoded;
  for(size_t i = 0; i < sentence.size(); ++i) {
    if(sentence[i] != EOS_ID || !ignoreEOS) {
      decoded.push_back((*this)[sentence[i]]);
    }
  }
  return decoded;
}


const std::string& Vocab::operator[](size_t id) const {
  amunmt_UTIL_THROW_IF2(id >= id2str_.size(), "Unknown word id: " << id);
  return id2str_[id];
}

size_t Vocab::size() const {
  return id2str_.size();
}

}

