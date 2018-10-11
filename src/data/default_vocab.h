#pragma once

#include "data/vocab_impl.h"

#include "3rd_party/exception.h"
#include "3rd_party/yaml-cpp/yaml.h"
#include "common/logging.h"
#include "common/regex.h"
#include "common/utils.h"
#include "common/filesystem.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace marian {

class DefaultVocab : public VocabImpl {
public:
  virtual int load(const std::string& vocabPath, int max = 0) override;
  virtual void create(const std::string& vocabPath, const std::string& trainPath) override;
  virtual void create(io::InputFileStream& trainStrm,
                      io::OutputFileStream& vocabStrm,
                      size_t maxSize = 0) override;
  
  virtual const std::string& canonicalSuffix() const { return suffixes_[0]; }
  virtual const std::vector<std::string>& suffixes() const { return suffixes_; }

  virtual Word operator[](const std::string& word) const override;

  virtual const std::string& operator[](Word id) const override;

  virtual Words encode(const std::string& line,
                       bool addEOS = true,
                       bool inference = false) const;

  virtual std::string decode(const Words& sentence,
                             bool ignoreEOS = true) const;

  virtual size_t size() const;

  virtual std::string type() const { return "DefaultVocab"; }

  virtual Word getEosId() const override { return eosId_; }
  virtual Word getUnkId() const override { return unkId_; }

  virtual void createFake();

private:
  virtual Words operator()(const std::vector<std::string>& lineTokens,
                          bool addEOS = true) const;

  virtual std::vector<std::string> operator()(const Words& sentence,
                                              bool ignoreEOS = true) const;

  Word insertWord(Word id, const std::string& str);

  typedef std::map<std::string, Word> Str2Id;
  Str2Id str2id_;

  typedef std::vector<std::string> Id2Str;
  Id2Str id2str_;

  Word eosId_ = (Word)-1;
  Word unkId_ = (Word)-1;

  std::vector<std::string> suffixes_ = { ".yml", ".yaml", ".json" };

  class VocabFreqOrderer;
};

Word DefaultVocab::operator[](const std::string& word) const {
  auto it = str2id_.find(word);
  if(it != str2id_.end())
    return it->second;
  else
    return unkId_;
}

Words DefaultVocab::operator()(const std::vector<std::string>& lineTokens,
                        bool addEOS) const {
  Words words(lineTokens.size());
  std::transform(lineTokens.begin(),
                 lineTokens.end(),
                 words.begin(),
                 [&](const std::string& w) { return (*this)[w]; });
  if(addEOS)
    words.push_back(eosId_);
  return words;
}

std::vector<std::string> DefaultVocab::operator()(const Words& sentence,
                                           bool ignoreEOS) const {
  std::vector<std::string> decoded;
  for(size_t i = 0; i < sentence.size(); ++i) {
    if((sentence[i] != eosId_ || !ignoreEOS)) {
      decoded.push_back((*this)[sentence[i]]);
    }
  }
  return decoded;
}

Words DefaultVocab::encode(const std::string& line, bool addEOS, bool /*inference*/) const {
  std::vector<std::string> lineTokens;
  utils::split(line, lineTokens, " ");
  return (*this)(lineTokens, addEOS);
}

std::string DefaultVocab::decode(const Words& sentence, bool ignoreEOS) const {
  std::string line;
  auto tokens = (*this)(sentence, ignoreEOS);
  return utils::join(tokens, " ");
}

const std::string& DefaultVocab::operator[](Word id) const {
  ABORT_IF(id >= id2str_.size(), "Unknown word id: ", id);
  return id2str_[id];
}

size_t DefaultVocab::size() const {
  return id2str_.size();
}

// helper to insert a word into str2id_[] and id2str_[]
Word DefaultVocab::insertWord(Word id, const std::string& str) {
  str2id_[str] = id;
  if(id >= id2str_.size())
    id2str_.resize(id + 1);
  id2str_[id] = str;
  return id;
};

int DefaultVocab::load(const std::string& vocabPath, int max) {
  bool isJson = regex::regex_search(vocabPath, regex::regex("\\.(json|yaml|yml)$"));
  LOG(info,
      "[data] Loading vocabulary from {} file {}",
      isJson ? "JSON/Yaml" : "text",
      vocabPath);
  ABORT_IF(!filesystem::exists(vocabPath),
           "DefaultVocabulary file {} does not exits",
           vocabPath);

  std::map<std::string, Word> vocab;
  // read from JSON (or Yaml) file
  if(isJson) {
    YAML::Node vocabNode = YAML::Load(io::InputFileStream(vocabPath));
    for(auto&& pair : vocabNode)
      vocab.insert({pair.first.as<std::string>(), pair.second.as<Word>()});
  }
  // read from flat text file
  else {
    io::InputFileStream in(vocabPath);
    std::string line;
    while(io::getline(in, line)) {
      ABORT_IF(line.empty(),
               "DefaultVocabulary file {} must not contain empty lines",
               vocabPath);
      vocab.insert({line, vocab.size()});
    }
    ABORT_IF(in.bad(), "DefaultVocabulary file {} could not be read", vocabPath);
  }

  std::unordered_set<Word> seenSpecial;

  id2str_.reserve(vocab.size());
  for(auto&& pair : vocab) {
    auto str = pair.first;
    auto id = pair.second;

    if(SPEC2SYM.count(str)) {
      seenSpecial.insert(id);
    }

    // note: this requires ids to be sorted by frequency
    if(!max || id < (Word)max) {
      insertWord(id, str);
    }
  }
  ABORT_IF(id2str_.empty(), "Empty vocabulary: ", vocabPath);

  // look up ids for </s> and <unk>, which are required
  // The name backCompatStr is alternatively accepted for Yaml vocabs if id
  // equals backCompatId.
  auto getRequiredWordId = [&](const std::string& str,
                               const std::string& backCompatStr,
                               Word backCompatId) {
    // back compat with Nematus Yaml dicts
    if(isJson) {
      // if word id 0 or 1 is either empty or has the Nematus-convention string,
      // then use it
      if(backCompatId < id2str_.size()
         && (id2str_[backCompatId].empty()
             || id2str_[backCompatId] == backCompatStr)) {
        LOG(info,
            "[data] Using unused word id {} for {}",
            backCompatStr,
            backCompatId,
            str);
        return backCompatId;
      }
    }
    auto iter = str2id_.find(str);
    ABORT_IF(iter == str2id_.end(),
             "DefaultVocabulary file {} is expected to contain an entry for {}",
             vocabPath,
             str);
    return iter->second;
  };
  eosId_ = getRequiredWordId(DEFAULT_EOS_STR, NEMATUS_EOS_STR, DEFAULT_EOS_ID);
  unkId_ = getRequiredWordId(DEFAULT_UNK_STR, NEMATUS_UNK_STR, DEFAULT_UNK_ID);

  // some special symbols for hard attention
  if(!seenSpecial.empty()) {
    auto requireWord = [&](Word id, const std::string& str) {
      auto iter = str2id_.find(str);
      // word already in vocab: must be at right index, else fail
      if(iter != str2id_.end())
        ABORT_IF(iter->second != id,
                 "special vocabulary entry '{}' is expected to have id {}",
                 str,
                 id);
      else
        insertWord(id, str);
    };
    // @TODO: the hard-att code has not yet been updated to accept EOS at any id
    requireWord(DEFAULT_EOS_ID, DEFAULT_EOS_STR);
    for(auto id : seenSpecial)
      requireWord(id, SYM2SPEC.at(id));
  }

  return std::max((int)id2str_.size(), max);
}

// for fakeBatch()
void DefaultVocab::createFake() {
  eosId_ = insertWord(DEFAULT_EOS_ID, DEFAULT_EOS_STR);
  unkId_ = insertWord(DEFAULT_UNK_ID, DEFAULT_UNK_STR);
}

class DefaultVocab::VocabFreqOrderer {
private:
  std::unordered_map<std::string, size_t>& counter_;

public:
  VocabFreqOrderer(std::unordered_map<std::string, size_t>& counter)
      : counter_(counter) {}

  bool operator()(const std::string& a, const std::string& b) const {
    return counter_[a] > counter_[b] || (counter_[a] == counter_[b] && a < b);
  }
};

void DefaultVocab::create(const std::string& vocabPath, const std::string& trainPath) {
  LOG(info, "[data] Creating vocabulary {} from {}", vocabPath, trainPath);

  filesystem::Path path(vocabPath);
  auto dir = path.parentPath();
  if(dir.empty())
    dir = filesystem::currentPath();

  ABORT_IF(!dir.empty() && !filesystem::isDirectory(dir),
           "Specified vocab directory {} does not exist",
           (std::string)dir);

  ABORT_IF(!dir.empty() && !filesystem::canWrite(dir),
           "No write permission in vocab directory {}",
           (std::string)dir);

  ABORT_IF(filesystem::exists(vocabPath),
           "DefaultVocab file '{}' exists. Not overwriting",
           (std::string)vocabPath);

  io::InputFileStream trainStrm(trainPath);
  io::OutputFileStream vocabStrm(vocabPath);
  create(trainStrm, vocabStrm);
}

void DefaultVocab::create(io::InputFileStream& trainStrm,
                   io::OutputFileStream& vocabStrm,
                   size_t maxSize) {
  std::string line;
  std::unordered_map<std::string, size_t> counter;

  std::unordered_set<Word> seenSpecial;

  while(getline((std::istream&)trainStrm, line)) {
    std::vector<std::string> toks;

    // we do not want any unexpected behavior during creation
    // e.g. sampling, hence use inference mode
    utils::split(line, toks, " ");

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
  vocabYaml.force_insert(DEFAULT_EOS_STR, DEFAULT_EOS_ID);
  vocabYaml.force_insert(DEFAULT_UNK_STR, DEFAULT_UNK_ID);

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

  vocabStrm << vocabYaml;
}

}