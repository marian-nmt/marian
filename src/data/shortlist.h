#pragma once

#include "common/config.h"
#include "common/definitions.h"
#include "common/file_stream.h"

#include <random>
#include <unordered_map>
#include <vector>
#include <iostream>

namespace marian {
namespace data {

class Shortlist {
private:
  std::vector<Word> indices_;
  std::vector<Word> mappedIndices_;
  std::vector<Word> reverseMap_;

public:
  Shortlist(const std::vector<Word>& indices,
            const std::vector<Word>& mappedIndices,
            const std::vector<Word>& reverseMap)
      : indices_(indices),
        mappedIndices_(mappedIndices),
        reverseMap_(reverseMap) {}

  std::vector<Word>& indices() { return indices_; }
  std::vector<Word>& mappedIndices() { return mappedIndices_; }
  Word reverseMap(Word idx) { return reverseMap_[idx]; }
};

class ShortlistGenerator {
public:
  virtual Ptr<Shortlist> generate(Ptr<data::CorpusBatch> batch) = 0;

  // Writes text version of (possibly) pruned short list to file
  // with given prefix and implementation-specific suffixes.
  virtual void dump(const std::string& /*prefix*/) {
    ABORT("Not implemented");
  }
};

class SampledShortlistGenerator : public ShortlistGenerator {
private:
  Ptr<Options> options_;
  size_t maxVocab_{50000};

  size_t total_{10000};
  size_t firstNum_{1000};

  size_t srcIdx_;
  size_t trgIdx_;
  bool shared_{false};

  std::random_device rd_;
  std::mt19937 gen_;

public:
  SampledShortlistGenerator(Ptr<Options> options,
                            size_t srcIdx = 0,
                            size_t trgIdx = 1,
                            bool shared = false)
      : options_(options),
        srcIdx_(srcIdx),
        trgIdx_(trgIdx),
        shared_(shared),
        gen_(rd_()) {}

  virtual Ptr<Shortlist> generate(Ptr<data::CorpusBatch> batch) override {
    auto srcBatch = (*batch)[srcIdx_];
    auto trgBatch = (*batch)[trgIdx_];

    // add firstNum most frequent words
    std::unordered_set<Word> idxSet;
    for(Word i = 0; i < firstNum_ && i < maxVocab_; ++i)
      idxSet.insert(i);

    // add all words from ground truth
    for(auto i : trgBatch->data())
      idxSet.insert(i);

    // add all words from source
    if(shared_)
      for(auto i : srcBatch->data())
        idxSet.insert(i);

    std::uniform_int_distribution<> dis((int)firstNum_, (int)maxVocab_);
    while(idxSet.size() < total_ && idxSet.size() < maxVocab_)
      idxSet.insert(dis(gen_));

    // turn into vector and sort (selected indices)
    std::vector<Word> idx(idxSet.begin(), idxSet.end());
    std::sort(idx.begin(), idx.end());

    // assign new shifted position
    std::unordered_map<Word, Word> pos;
    std::vector<Word> reverseMap;

    for(Word i = 0; i < idx.size(); ++i) {
      pos[idx[i]] = i;
      reverseMap.push_back(idx[i]);
    }

    std::vector<Word> mapped;
    for(auto i : trgBatch->data()) {
      // mapped postions for cross-entropy
      mapped.push_back(pos[i]);
    }

    return New<Shortlist>(idx, mapped, reverseMap);
  }
};

class LexicalShortlistGenerator : public ShortlistGenerator {
private:
  Ptr<Options> options_;
  Ptr<Vocab> srcVocab_;
  Ptr<Vocab> trgVocab_;

  size_t srcIdx_;
  size_t trgIdx_;
  bool shared_{false};

  size_t firstNum_{100};
  size_t bestNum_{100};

  std::vector<std::unordered_map<Word, float>> data_;

  void load(const std::string& fname) {
    io::InputFileStream in(fname);

    std::string src, trg;
    float prob;
    while(in >> trg >> src >> prob) {
      // @TODO: change this to something safer other than NULL
      if(src == "NULL" || trg == "NULL")
        continue;

      Word sId = (*srcVocab_)[src];
      Word tId = (*trgVocab_)[trg];

      if(data_.size() <= sId)
        data_.resize(sId + 1);
      data_[sId][tId] = prob;
    }
  }

  void prune(float threshold = 0.f) {
    size_t i = 0;
    for(auto& probs : data_) {
      std::vector<std::pair<float, Word>> sorter;
      for(auto& it : probs)
        sorter.emplace_back(it.second, (Word)it.first);

      std::sort(
          sorter.begin(), sorter.end(), std::greater<std::pair<float, Word>>());

      probs.clear();
      for(auto& it : sorter) {
        if(probs.size() < bestNum_ && it.first > threshold)
          probs[it.second] = it.first;
        else
          break;
      }

      ++i;
    }
  }

public:
  LexicalShortlistGenerator(Ptr<Options> options,
                            Ptr<Vocab> srcVocab,
                            Ptr<Vocab> trgVocab,
                            size_t srcIdx = 0,
                            size_t trgIdx = 1,
                            bool shared = false)
      : options_(options),
        srcVocab_(srcVocab),
        trgVocab_(trgVocab),
        srcIdx_(srcIdx),
        trgIdx_(trgIdx),
        shared_(shared) {
    std::vector<std::string> vals = options_->get<std::vector<std::string>>("shortlist");

    ABORT_IF(vals.empty(), "No path to filter path given");
    std::string fname = vals[0];

    firstNum_ = vals.size() > 1 ? std::stoi(vals[1]) : 100;
    bestNum_ = vals.size() > 2 ? std::stoi(vals[2]) : 100;
    float threshold = vals.size() > 3 ? std::stof(vals[3]) : 0;
    std::string dumpPath = vals.size() > 4 ? vals[4] : "";

    LOG(info,
        "[data] Loading lexical shortlist as {} {} {} {}",
        fname,
        firstNum_,
        bestNum_,
        threshold);

    load(fname);
    prune(threshold);

    if(!dumpPath.empty())
      dump(dumpPath);
  }

  virtual void dump(const std::string& prefix) override {
    // Dump top most frequent words from target vocabulary
    LOG(info, "[data] Saving shortlist dump to {}", prefix + ".{top,dic}");
    io::OutputFileStream outTop(prefix + ".top");
    for(Word i = 0; i < firstNum_ && i < trgVocab_->size(); ++i)
      outTop << (*trgVocab_)[i] << std::endl;

    // Dump translation pairs from dictionary
    io::OutputFileStream outDic(prefix + ".dic");
    for(Word srcId = 0; srcId < data_.size(); srcId++) {
      for(auto& it : data_[srcId]) { // @TODO: change data_.first from size_t to Word
        Word trgId = (Word)it.first;
        outDic << (*srcVocab_)[srcId] << "\t" << (*trgVocab_)[trgId] << std::endl;
      }
    }
  }

  virtual Ptr<Shortlist> generate(Ptr<data::CorpusBatch> batch) override {
    auto srcBatch = (*batch)[srcIdx_];
    // auto trgBatch = (*batch)[trgIdx_];

    // add firstNum most frequent words
    std::unordered_set<Word> idxSet;
    for(Word i = 0; i < firstNum_ && i < trgVocab_->size(); ++i)
      idxSet.insert(i);

    // add all words from ground truth
    // for(auto i : trgBatch->data())
    //  idxSet.insert(i);

    // collect unique words form source
    std::unordered_set<Word> srcSet;
    for(auto i : srcBatch->data())
      srcSet.insert(i);

    // add aligned target words
    for(auto i : srcSet) {
      if(shared_)
        idxSet.insert(i);
      for(auto& it : data_[i])
        idxSet.insert((Word)it.first); // @TODO: change it.first to Word
    }

    // turn into vector and sort (selected indices)
    std::vector<Word> idx(idxSet.begin(), idxSet.end());
    std::sort(idx.begin(), idx.end());

    // assign new shifted position
    // std::unordered_map<Word, Word> pos;
    std::vector<Word> reverseMap;

    for(Word i = 0; i < idx.size(); ++i) {
      // pos[idx[i]] = i;
      reverseMap.push_back(idx[i]);
    }

    std::vector<Word> mapped;
    // for(auto i : trgBatch->data()) {
    // mapped postions for cross-entropy
    // mapped.push_back(pos[i]);
    //}

    return New<Shortlist>(idx, mapped, reverseMap);
  }
};

class FakeShortlistGenerator : public ShortlistGenerator {
private:
  std::vector<Word> idx_;
  std::vector<Word> reverseIdx_;

public:
  FakeShortlistGenerator(const std::unordered_set<Word>& idxSet)
      : idx_(idxSet.begin(), idxSet.end()) {
    std::sort(idx_.begin(), idx_.end());
    // assign new shifted position
    for(Word i = 0; i < idx_.size(); ++i) {
      reverseIdx_.push_back(idx_[i]);
    }
  }

  Ptr<Shortlist> generate(Ptr<data::CorpusBatch> /*batch*/) override {
    std::vector<Word> tmp;
    return New<Shortlist>(idx_, tmp, reverseIdx_);
  }
};

}  // namespace data
}  // namespace marian
