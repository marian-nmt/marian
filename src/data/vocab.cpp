#include <algorithm>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "3rd_party/exception.h"
#include "3rd_party/yaml-cpp/yaml.h"
#include "common/logging.h"
#include "common/utils.h"
#include "data/vocab.h"

namespace marian {

Vocab::Vocab() {}

size_t Vocab::operator[](const std::string& word) const {
  auto it = str2id_.find(word);
  if(it != str2id_.end())
    return it->second;
  else
    return UNK_ID;
}

Words Vocab::operator()(const std::vector<std::string>& lineTokens,
                        bool addEOS) const {
  Words words(lineTokens.size());
  std::transform(lineTokens.begin(),
                 lineTokens.end(),
                 words.begin(),
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

std::vector<std::string> Vocab::operator()(const Words& sentence,
                                           bool ignoreEOS) const {
  std::vector<std::string> decoded;
  for(size_t i = 0; i < sentence.size(); ++i) {
    if((sentence[i] != EOS_ID || !ignoreEOS)) {
      decoded.push_back((*this)[sentence[i]]);
    }
  }
  return decoded;
}

const std::string& Vocab::operator[](size_t id) const {
  ABORT_IF(id >= id2str_.size(), "Unknown word id: ", id);
  return id2str_[id];
}

size_t Vocab::size() const {
  return id2str_.size();
}

int Vocab::loadOrCreate(const std::string& vocabPath,
                        const std::string& trainPath,
                        int max) {
  if(vocabPath.empty()) {
    if(boost::filesystem::exists(trainPath + ".json")) {
      return load(trainPath + ".json", max);
    }
    if(boost::filesystem::exists(trainPath + ".yml")) {
      return load(trainPath + ".yml", max);
    }
    create(trainPath + ".yml", trainPath);
    return load(trainPath + ".yml", max);
  } else {
    if(!boost::filesystem::exists(vocabPath))
      create(vocabPath, trainPath);
    return load(vocabPath, max);
  }
}

int Vocab::load(const std::string& vocabPath, int max) {
  LOG(info, "[data] Loading vocabulary from {}", vocabPath);
  ABORT_IF(!boost::filesystem::exists(vocabPath),
           "Vocabulary {} does not exits",
           vocabPath);

  YAML::Node vocab = YAML::Load(InputFileStream(vocabPath));

  std::unordered_set<Word> seenSpecial;

  for(auto&& pair : vocab) {
    auto str = pair.first.as<std::string>();
    auto id = pair.second.as<Word>();

    if(SPEC2SYM.count(str)) {
      seenSpecial.insert(id);
    }

    if(!max || id < (Word)max) {
      str2id_[str] = id;
      if(id >= id2str_.size())
        id2str_.resize(id + 1);
      id2str_[id] = str;
    }
  }
  ABORT_IF(id2str_.empty(), "Empty vocabulary: ", vocabPath);

  id2str_[EOS_ID] = EOS_STR;
  id2str_[UNK_ID] = UNK_STR;
  for(auto id : seenSpecial)
    id2str_[id] = SYM2SPEC.at(id);

  return std::max((int)id2str_.size(), max);
}

class Vocab::VocabFreqOrderer {
private:
  std::unordered_map<std::string, size_t>& counter_;

public:
  VocabFreqOrderer(std::unordered_map<std::string, size_t>& counter)
      : counter_(counter) {}

  bool operator()(const std::string& a, const std::string& b) const {
    return counter_[a] > counter_[b] || (counter_[a] == counter_[b] && a < b);
  }
};

void Vocab::create(const std::string& vocabPath, const std::string& trainPath) {
  LOG(info, "[data] Creating vocabulary {} from {}", vocabPath, trainPath);

  boost::filesystem::path path(vocabPath);
  auto dir = path.parent_path();
  if(dir.empty())
    dir = boost::filesystem::current_path();

  ABORT_IF(!dir.empty() && !boost::filesystem::is_directory(dir),
           "Specified vocab directory {} does not exist",
           dir);

  ABORT_IF(!dir.empty()
               && !(boost::filesystem::status(dir).permissions()
                    & boost::filesystem::owner_write),
           "No write permission in vocab directory {}",
           dir);

  ABORT_IF(boost::filesystem::exists(vocabPath),
           "Vocab file '{}' exists. Not overwriting",
           vocabPath);

  InputFileStream trainStrm(trainPath);
  OutputFileStream vocabStrm(vocabPath);
  create(trainStrm, vocabStrm);
}

void Vocab::create(InputFileStream& trainStrm,
                   OutputFileStream& vocabStrm,
                   size_t maxSize) {
  std::string line;
  std::unordered_map<std::string, size_t> counter;

  std::unordered_set<Word> seenSpecial;

  while(getline((std::istream&)trainStrm, line)) {
    std::vector<std::string> toks;
    Split(line, toks);

    for(const std::string& tok : toks) {
      if(SPEC2SYM.count(tok)) {
        seenSpecial.insert(SPEC2SYM.at(tok));
        continue;
      }

      auto iter = counter.find(tok);
      if(iter == counter.end())
        counter[tok] = 1;
      else
        iter->second++;
    }
  }

  std::vector<std::string> vocabVec;
  for(auto& p : counter)
    vocabVec.push_back(p.first);

  std::sort(vocabVec.begin(), vocabVec.end(), VocabFreqOrderer(counter));

  YAML::Node vocabYaml;
  vocabYaml.force_insert(EOS_STR, EOS_ID);
  vocabYaml.force_insert(UNK_STR, UNK_ID);

  for(auto word : seenSpecial)
    vocabYaml.force_insert(SYM2SPEC.at(word), word);

  Word maxSpec = 1;
  for(auto i : seenSpecial)
    if(i > maxSpec)
      maxSpec = i;

  auto vocabSize = vocabVec.size();
  if(maxSize > maxSpec)
    vocabSize = std::min(maxSize - maxSpec - 1, vocabVec.size());

  for(size_t i = 0; i < vocabSize; ++i)
    vocabYaml.force_insert(vocabVec[i], i + maxSpec + 1);

  (std::ostream&)vocabStrm << vocabYaml;
}
}
