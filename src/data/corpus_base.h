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

/**
 * @brief Batch of sentences represented as word indices with masking.
 */
class SubBatch {
private:
  std::vector<Word> indices_;
  std::vector<float> mask_;

  size_t size_;
  size_t width_;
  size_t words_;

public:
  /**
   * @brief Creates an empty subbatch of specified size.
   *
   * @param size Number of sentences
   * @param width Number of words in the longest sentence
   */
  SubBatch(int size, int width)
      : indices_(size * width, 0),
        mask_(size * width, 0),
        size_(size),
        width_(width),
        words_(0) {}

  /**
   * @brief Flat vector of word indices.
   *
   * The order of indices is \f$idx_{0,0}, idx_{0,1},\dots,idx_{0,s}, \dots,
   * idx_{w,0},idx_{w,1},\dots,idx_{w,s}\f$, where \f$w\f$ is the number of
   * words (width) and \f$s\f$ is the number of sentences (size).
   */
  std::vector<Word>& indices() { return indices_; }
  /**
   * @brief Flat masking vector; 0 is used for masked words.
   *
   * @see indices()
   */
  std::vector<float>& mask() { return mask_; }

  /**
   * @brief The number of sentences in the batch.
   */
  size_t batchSize() { return size_; }
  /**
   * @brief The number of words in the longest sentence in the batch.
   */
  size_t batchWidth() { return width_; };
  /**
   * @brief The total number of words in the batch, including masking.
   */
  size_t batchWords() { return words_; }

  std::vector<Ptr<SubBatch>> split(size_t n) {
    std::vector<Ptr<SubBatch>> splits;

    size_t subSize = std::ceil(size_ / (float)n);
    size_t totSize = size_;

    int pos = 0;
    for(int k = 0; k < n; ++k) {
      size_t __size__ = std::min(subSize, totSize);

      auto sb = New<SubBatch>(__size__, width_);

      size_t __words__ = 0;
      for(int j = 0; j < width_; ++j) {
        for(int i = 0; i < __size__; ++i) {
          sb->indices()[j * __size__ + i] = indices_[j * size_ + pos + i];
          sb->mask()[j * __size__ + i] = mask_[j * size_ + pos + i];
          if(mask_[j * size_ + pos + i] != 0)
            __words__++;
        }
      }

      sb->setWords(__words__);
      splits.push_back(sb);

      totSize -= __size__;
      pos += __size__;
    }
    return splits;
  }

  void setWords(size_t words) { words_ = words; }
};

class CorpusBatch : public Batch {
private:
  std::vector<Ptr<SubBatch>> batches_;
  std::vector<float> guidedAlignment_;
  std::vector<float> dataWeights_;

public:
  CorpusBatch(const std::vector<Ptr<SubBatch>>& batches) : batches_(batches) {}

  Ptr<SubBatch> operator[](size_t i) const { return batches_[i]; }

  Ptr<SubBatch> front() { return batches_.front(); }

  Ptr<SubBatch> back() { return batches_.back(); }

  void debug() {
    std::cerr << "batches: " << sets() << std::endl;

    if(!sentenceIds_.empty()) {
      std::cerr << "indexes: ";
      for(auto id : sentenceIds_)
        std::cerr << id << " ";
      std::cerr << std::endl;
    }

    size_t b = 0;
    for(auto sb : batches_) {
      std::cerr << "batch " << b++ << ": " << std::endl;
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

  std::vector<Ptr<Batch>> split(size_t n) {
    std::vector<Ptr<Batch>> splits;

    std::vector<std::vector<Ptr<SubBatch>>> subs(n);

    for(auto subBatch : batches_) {
      size_t i = 0;
      for(auto splitSubBatch : subBatch->split(n))
        subs[i++].push_back(splitSubBatch);
    }

    for(auto subBatches : subs)
      splits.push_back(New<CorpusBatch>(subBatches));

    size_t pos = 0;
    for(auto split : splits) {
      std::vector<size_t> ids;
      for(int i = pos; i < pos + split->size(); ++i)
        ids.push_back(sentenceIds_[i]);
      split->setSentenceIds(ids);
      pos += split->size();
    }

    return splits;
  }

  size_t size() const { return batches_[0]->batchSize(); }

  size_t words() const { return batches_[0]->batchWords(); }

  size_t sets() const { return batches_.size(); }

  static Ptr<CorpusBatch> fakeBatch(std::vector<size_t>& lengths,
                                    size_t batchSize,
                                    bool guidedAlignment = false,
                                    bool dataWeights = false,
                                    // @TODO: is this needed here?
                                    bool sentenceLevel = false) {
    std::vector<Ptr<SubBatch>> batches;

    for(auto len : lengths) {
      auto sb = New<SubBatch>(batchSize, len);
      std::fill(sb->mask().begin(), sb->mask().end(), 1);

      batches.push_back(sb);
    }

    auto batch = New<CorpusBatch>(batches);

    if(guidedAlignment) {
      std::vector<float> guided(batchSize * lengths.front() * lengths.back(),
                                0.f);
      batch->setGuidedAlignment(guided);
    }

    if(dataWeights) {
      int weightsSize = batchSize;
      if(!sentenceLevel)
        weightsSize *= lengths.back();
      std::vector<float> weights(weightsSize, 0.f);
      batch->setDataWeights(weights);
    }

    return batch;
  }

  std::vector<float>& getGuidedAlignment() { return guidedAlignment_; }
  void setGuidedAlignment(const std::vector<float>& aln) {
    guidedAlignment_ = aln;
  }

  std::vector<float>& getDataWeights() { return dataWeights_; }
  void setDataWeights(const std::vector<float>& aln) {
    dataWeights_ = aln;
  }
};

class CorpusIterator;

class CorpusBase
    : public DatasetBase<SentenceTuple, CorpusIterator, CorpusBatch> {
public:
  CorpusBase() : DatasetBase() {}
  CorpusBase(Ptr<Config> options, bool translate = false);
  CorpusBase(std::vector<std::string> paths,
             std::vector<Ptr<Vocab>> vocabs,
             Ptr<Config> options,
             size_t maxLength);

  virtual std::vector<Ptr<Vocab>>& getVocabs() = 0;

protected:
  std::vector<UPtr<InputFileStream>> files_;
  std::vector<Ptr<Vocab>> vocabs_;

  Ptr<Config> options_;

  size_t maxLength_;
  bool maxLengthCrop_;
  bool rightLeft_;
};

class CorpusIterator
    : public boost::iterator_facade<CorpusIterator,
                                    SentenceTuple const,
                                    boost::forward_traversal_tag> {
public:
  CorpusIterator();
  explicit CorpusIterator(CorpusBase* corpus);

private:
  friend class boost::iterator_core_access;

  void increment();

  bool equal(CorpusIterator const& other) const;

  const SentenceTuple& dereference() const;

  CorpusBase* corpus_;

  long long int pos_;
  SentenceTuple tup_;
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

    LOG(info, "[data] Loading word alignment from {}", fname);

    while(std::getline((std::istream&)aStream, line)) {
      data_.emplace_back();
      std::vector<std::string> atok = split(line, " -");
      for(size_t i = 0; i < atok.size(); i += 2)
        data_.back().emplace_back(std::stoi(atok[i]), std::stoi(atok[i + 1]));
      c++;
    }

    LOG(info, "[data] Done");
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

class DataWeights {
private:
  std::vector<std::vector<float>> data_;
  bool sentLvl_;

public:
  DataWeights(const std::string& fname, bool sentLvl) : sentLvl_(sentLvl) {
    InputFileStream aStream(fname);
    std::string line;

    LOG(info, "[data] Loading weights from {}", fname);

    while(std::getline((std::istream&)aStream, line)) {
      data_.emplace_back();
      std::vector<std::string> weights;
      Split(line, weights, " ");
      for(auto w : weights)
        data_.back().emplace_back(std::stoi(w));
    }

    LOG(info, "[data] Done");
  }

  void weightsForBatch(Ptr<CorpusBatch> batch) {
    int dimBatch = batch->getSentenceIds().size();
    int trgWords = batch->back()->batchWidth();

    int s = sentLvl_ ? dimBatch : dimBatch * trgWords;
    std::vector<float> weights(s, 1.f);

    if(sentLvl_) {
      for(int b = 0; b < dimBatch; ++b)
        weights[b] = data_[batch->getSentenceIds()[b]].front();
    } else {
      for(int b = 0; b < dimBatch; ++b) {
        auto& sentWeights = data_[batch->getSentenceIds()[b]];
        size_t i = 0;
        for(auto& w : sentWeights) {
          weights[b + i * dimBatch] = w;
          ++i;
        }
      }
    }

    batch->setDataWeights(weights);
  }
};
}
}
