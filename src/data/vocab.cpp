
#include <sstream>
#include <algorithm>

#include "data/vocab.h"
#include "common/utils.h"
#include "common/file_stream.h"
#include "3rd_party/exception.h"
#include "3rd_party/yaml-cpp/yaml.h"
#include "common/logging.h"

Vocab::Vocab() {
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
  UTIL_THROW_IF2(id >= id2str_.size(), "Unknown word id: " << id);
  return id2str_[id];
}

size_t Vocab::size() const {
  return id2str_.size();
}

void Vocab::loadOrCreate(const std::string& trainPath, int max)
{
  if(boost::filesystem::exists(trainPath + ".json")) {
    load(trainPath + ".json", max);
    return;
  }
  if(boost::filesystem::exists(trainPath + ".yml")) {
    load(trainPath + ".yml", max);
    return;
  }

  create(trainPath + ".yml", max, trainPath);
  load(trainPath + ".yml", max);
}

void Vocab::load(const std::string& vocabPath, int max)
{
  LOG(data, "Loading vocabulary from {} (max: {})", vocabPath, max);
  YAML::Node vocab = YAML::Load(InputFileStream(vocabPath));
  for(auto&& pair : vocab) {
    auto str = pair.first.as<std::string>();
    auto id = pair.second.as<Word>();
    if (id < (Word)max) {
      str2id_[str] = id;
      if(id >= id2str_.size())
        id2str_.resize(id + 1);
      id2str_[id] = str;
    }
  }
  UTIL_THROW_IF2(id2str_.empty(), "Empty vocabulary " << vocabPath);

  id2str_[EOS_ID] = EOS_STR;
  id2str_[UNK_ID] = UNK_STR;
}

class Vocab::VocabFreqOrderer
{
public:
  bool operator()(const Vocab::Str2Id::value_type* a, const Vocab::Str2Id::value_type* b) const {
    return a->second < b->second;
  }
};

void Vocab::create(const std::string& vocabPath, int max, const std::string& trainPath)
{
  LOG(data,"Creating vocabulary {} from {} (max: {})", vocabPath, trainPath, max);

  UTIL_THROW_IF2(boost::filesystem::exists(vocabPath),
                 "Vocab file " << vocabPath << " exists. Not overwriting");

  InputFileStream trainStrm(trainPath);

  Str2Id vocab;
  std::string line;
  while (getline((std::istream&)trainStrm, line)) {
    std::vector<std::string> toks;
    Split(line, toks);

    for (const std::string &tok: toks) {
      Str2Id::iterator iter = vocab.find(tok);
      if (iter == vocab.end())
        vocab[tok] = 1;
      else
        iter->second++;
    }
  }

  // put into vector & sort
  std::vector<const Str2Id::value_type*> vocabVec;
  vocabVec.reserve(max);

  for (const Str2Id::value_type &p: vocab)
    vocabVec.push_back(&p);
  std::sort(vocabVec.rbegin(), vocabVec.rend(), VocabFreqOrderer());

  YAML::Node vocabYaml;
  vocabYaml[EOS_STR] = EOS_ID;
  vocabYaml[UNK_STR] = UNK_ID;
  for(size_t i = 0; i < vocabVec.size(); ++i) {
    const Str2Id::value_type *p = vocabVec[i];
    vocabYaml[p->first] = i + 2;
  }

  OutputFileStream vocabStrm(vocabPath);
  (std::ostream&)vocabStrm << vocabYaml;
}
