#pragma once

#include <fstream>
#include <iostream>
#include <random>

#include <boost/algorithm/string.hpp>
#include <boost/iterator/iterator_facade.hpp>

#include "common/config.h"
#include "common/definitions.h"
#include "common/file_stream.h"
#include "data/batch.h"
#include "data/dataset.h"
#include "data/vocab.h"

namespace marian {
namespace data {

class SentenceTuple {
private:
  size_t id_;
  std::vector<Words> tuple_;

public:
  SentenceTuple(size_t id) : id_(id) {}

  ~SentenceTuple() { tuple_.clear(); }

  void push_back(const Words& words) { tuple_.push_back(words); }

  size_t size() const { return tuple_.size(); }

  Words& operator[](size_t i) { return tuple_[i]; }

  Words& back() { return tuple_.back(); }
  const Words& back() const { return tuple_.back(); }

  const Words& operator[](size_t i) const { return tuple_[i]; }

  bool empty() const { return tuple_.empty(); }

  auto begin() -> decltype(tuple_.begin()) { return tuple_.begin(); }
  auto end() -> decltype(tuple_.end()) { return tuple_.end(); }

  size_t getId() const { return id_; }
};

class SubBatch {
private:
  std::vector<Word> indices_;
  std::vector<float> mask_;

  size_t size_;
  size_t width_;
  size_t words_;

public:
  SubBatch(int size, int width)
      : indices_(size * width, 0),
        mask_(size * width, 0),
        size_(size),
        width_(width),
        words_(0) {}

  std::vector<Word>& indices() { return indices_; }
  std::vector<float>& mask() { return mask_; }

  size_t batchSize() { return size_; }
  size_t batchWidth() { return width_; };
  size_t batchWords() { return words_; }

  void setWords(size_t words) { words_ = words; }
};

class CorpusBatch : public Batch {
private:
  std::vector<Ptr<SubBatch>> batches_;
  std::vector<size_t> sentenceIds_;
  std::vector<float> guidedAlignment_;

public:
  CorpusBatch(const std::vector<Ptr<SubBatch>>& batches) : batches_(batches) {}

  Ptr<SubBatch> operator[](size_t i) const { return batches_[i]; }

  Ptr<SubBatch> front() { return batches_.front(); }

  Ptr<SubBatch> back() { return batches_.back(); }

  void debug() {
    size_t i = 0;
    if(!sentenceIds_.empty()) {
      for(auto i : sentenceIds_)
        std::cerr << i << " ";
      std::cerr << std::endl;
    }

    for(auto sb : batches_) {
      std::cerr << "input " << i++ << ": " << std::endl;
      for(size_t i = 0; i < sb->batchWidth(); i++) {
        std::cerr << "\t w: ";
        for(size_t j = 0; j < sb->batchSize(); j++) {
          Word w = sb->indices()[i * sb->batchSize() + j];
          std::cerr << w << " ";
        }
        std::cerr << std::endl;
      }
    }
  }

  size_t size() const { return batches_[0]->batchSize(); }

  size_t words() const { return batches_[0]->batchWords(); }

  size_t sets() const { return batches_.size(); }

  static Ptr<CorpusBatch> fakeBatch(std::vector<size_t>& lengths,
                                    size_t batchSize,
                                    bool guidedAlignment = false) {
    std::vector<Ptr<SubBatch>> batches;

    for(auto len : lengths) {
      auto sb = New<SubBatch>(batchSize, len);
      batches.push_back(sb);
    }

    auto batch = New<CorpusBatch>(batches);

    if(guidedAlignment) {
      std::vector<float> guided(batchSize * lengths.front() * lengths.back(),
                                0.f);
      batch->setGuidedAlignment(guided);
    }

    return batch;
  }

  std::vector<float>& getGuidedAlignment() { return guidedAlignment_; }

  void setGuidedAlignment(const std::vector<float>& aln) {
    guidedAlignment_ = aln;
  }
};

class Corpus;

class CorpusIterator
    : public boost::iterator_facade<CorpusIterator,
                                    SentenceTuple const,
                                    boost::forward_traversal_tag> {
public:
  CorpusIterator();
  explicit CorpusIterator(Corpus& corpus);

private:
  friend class boost::iterator_core_access;

  void increment();

  bool equal(CorpusIterator const& other) const;

  const SentenceTuple& dereference() const;

  Corpus* corpus_;

  SentenceTuple tup_;
  long long int pos_;
};

class WordAlignment {
private:
  typedef std::pair<int, int> Point;
  typedef std::vector<Point> Alignment;

  std::vector<Alignment> data_;

public:
  WordAlignment(const std::string& fname) {
    InputFileStream aStream(fname);
    std::string line;
    size_t c = 0;

    LOG(data)->info("Loading word alignment from {}", fname);

    while(std::getline((std::istream&)aStream, line)) {
      data_.emplace_back();
      std::vector<std::string> atok = split(line, " -");
      ;
      for(size_t i = 0; i < atok.size(); i += 2)
        data_.back().emplace_back(std::stoi(atok[i]), std::stoi(atok[i + 1]));
      c++;
    }

    LOG(data)->info("Done");
  }

  std::vector<std::string> split(const std::string& input,
                                 const std::string& chars) {
    std::vector<std::string> output;
    boost::split(output, input, boost::is_any_of(chars));
    return output;
  }

  void guidedAlignment(Ptr<CorpusBatch> batch) {
    int srcWords = batch->front()->batchWidth();
    int trgWords = batch->back()->batchWidth();

    int dimBatch = batch->getSentenceIds().size();
    std::vector<float> guided(dimBatch * srcWords * trgWords, 0.f);

    for(int b = 0; b < dimBatch; ++b) {
      auto& alignment = data_[batch->getSentenceIds()[b]];
      for(auto& p : alignment) {
        int sid, tid;
        std::tie(sid, tid) = p;

        size_t idx = b + sid * dimBatch + tid * srcWords * dimBatch;
        guided[idx] = 1.f;
      }
    }
    batch->setGuidedAlignment(guided);
  }
};

class Corpus : public DatasetBase<SentenceTuple, CorpusIterator, CorpusBatch> {
private:
  Ptr<Config> options_;

  std::vector<UPtr<TemporaryFile>> tempFiles_;
  std::vector<UPtr<InputFileStream>> files_;
  std::vector<Ptr<Vocab>> vocabs_;
  size_t maxLength_;

  std::mt19937 g_;
  std::vector<size_t> ids_;
  size_t pos_{0};

  Ptr<WordAlignment> wordAlignment_;

  void shuffleFiles(const std::vector<std::string>& paths);

public:
  Corpus(Ptr<Config> options, bool translate = false);

  Corpus(std::vector<std::string> paths,
         std::vector<Ptr<Vocab>> vocabs,
         Ptr<Config> options,
         size_t maxLength = 0);

  /**
   * @brief Iterates sentence tuples in the corpus.
   *
   * A sentence tuple is skipped with no warning if any sentence in the tuple
   * (e.g. a source or target) is longer than the maximum allowed sentence
   * length in words.
   *
   * @return A tuple representing parallel sentences.
   */
  sample next();

  void shuffle();

  void reset();

  iterator begin() { return iterator(*this); }

  iterator end() { return iterator(); }

  std::vector<Ptr<Vocab>>& getVocabs() { return vocabs_; }

  batch_ptr toBatch(const std::vector<sample>& batchVector) {
    int batchSize = batchVector.size();

    std::vector<size_t> sentenceIds;

    std::vector<int> maxDims;
    for(auto& ex : batchVector) {
      if(maxDims.size() < ex.size())
        maxDims.resize(ex.size(), 0);
      for(size_t i = 0; i < ex.size(); ++i) {
        if(ex[i].size() > (size_t)maxDims[i])
          maxDims[i] = ex[i].size();
      }
      sentenceIds.push_back(ex.getId());
    }

    std::vector<Ptr<SubBatch>> subBatches;
    for(auto m : maxDims) {
      subBatches.emplace_back(New<SubBatch>(batchSize, m));
    }

    std::vector<size_t> words(maxDims.size(), 0);
    for(int i = 0; i < batchSize; ++i) {
      for(int j = 0; j < maxDims.size(); ++j) {
        for(int k = 0; k < batchVector[i][j].size(); ++k) {
          subBatches[j]->indices()[k * batchSize + i] = batchVector[i][j][k];
          subBatches[j]->mask()[k * batchSize + i] = 1.f;
          words[j]++;
        }
      }
    }

    for(size_t j = 0; j < maxDims.size(); ++j)
      subBatches[j]->setWords(words[j]);

    auto batch = batch_ptr(new batch_type(subBatches));
    batch->setSentenceIds(sentenceIds);

    if(options_->has("guided-alignment") && wordAlignment_)
      wordAlignment_->guidedAlignment(batch);

    return batch;
  }

  void prepare() {
    if(options_->has("guided-alignment"))
      setWordAlignment(options_->get<std::string>("guided-alignment"));
  }

private:
  void setWordAlignment(const std::string& path) {
    wordAlignment_ = New<WordAlignment>(path);
  }
};
}
}
