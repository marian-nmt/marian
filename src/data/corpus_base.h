#pragma once

#include <fstream>
#include <iostream>
#include <random>

#include <boost/algorithm/string.hpp>
#include <boost/iterator/iterator_facade.hpp>

#include "common/config.h"
#include "common/definitions.h"
#include "common/file_stream.h"
#include "common/options.h"
#include "data/alignment.h"
#include "data/batch.h"
#include "data/dataset.h"
#include "data/rng_engine.h"
#include "data/vocab.h"

namespace marian {
namespace data {

/**
 * @brief A sentence tuple that stores all sources and target sentences for a
 * specific "line" from a parallel corpus.
 *
 * Sentence tuples are used to store sentences read from external files and to
 * be a basis for construction of marian::data::CorpusBatch objects. They are
 * not a part of marian::data::CorpusBatch.
 */
class SentenceTuple {
private:
  size_t id_;
  std::vector<Words> tuple_;
  std::vector<float> weights_;
  WordAlignment alignment_;

public:
  typedef Words value_type;

  /**
   * @brief Creates an empty tuple with the given Id.
   */
  SentenceTuple(size_t id) : id_(id) {}

  ~SentenceTuple() { tuple_.clear(); }

  /**
   * @brief Returns the sentence's ID.
   */
  size_t getId() const { return id_; }

  /**
   * @brief Adds a new sentence at the end of the tuple.
   *
   * @param words A vector of word indexes.
   */
  void push_back(const Words& words) { tuple_.push_back(words); }

  /**
   * @brief The size of the tuple, e.g. two for parallel data with a source and
   * target sentences.
   */
  size_t size() const { return tuple_.size(); }

  /**
   * @brief The i-th tuple sentence.
   *
   * @param i Tuple's index.
   */
  Words& operator[](size_t i) { return tuple_[i]; }
  const Words& operator[](size_t i) const { return tuple_[i]; }

  /**
   * @brief The last tuple sentence, i.e. the target sentence.
   */
  Words& back() { return tuple_.back(); }
  const Words& back() const { return tuple_.back(); }

  /**
   * @brief Checks whether the tuple is empty.
   */
  bool empty() const { return tuple_.empty(); }

  auto begin() const -> decltype(tuple_.begin()) { return tuple_.begin(); }
  auto end() const -> decltype(tuple_.end()) { return tuple_.end(); }

  auto rbegin() const -> decltype(tuple_.rbegin()) { return tuple_.rbegin(); }
  auto rend() const -> decltype(tuple_.rend()) { return tuple_.rend(); }

  /**
   * @brief  Get sentence weights.
   *
   * For sentence-level weights the vector contains only one element.
   */
  const std::vector<float>& getWeights() const { return weights_; }
  void setWeights(const std::vector<float>& weights) { weights_ = weights; }

  const WordAlignment& getAlignment() const { return alignment_; }
  void setAlignment(const WordAlignment& alignment) { alignment_ = alignment; }
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
  std::vector<Word>& data() { return indices_; }
  /**
   * @brief Flat masking vector; 0 is used for masked words.
   *
   * @see data()
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
   * @brief The total number of words in the batch, considering the mask.
   */
  size_t batchWords() { return words_; }

  /**
   * @brief Splits the subbatch into subbatches of equal size.
   *
   * @param n Number of splits
   *
   * @return Vector of pointers to new subbatches.
   *
   * @see marian::data::Batch::split(size_t n)
   */
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
          sb->data()[j * __size__ + i] = indices_[j * size_ + pos + i];
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

/**
 * @brief Batch of source and target sentences with additional information,
 * such as guided alignments and sentence or word-leve weighting.
 */
class CorpusBatch : public Batch {
private:
  std::vector<Ptr<SubBatch>> batches_;
  std::vector<float> guidedAlignment_;
  std::vector<float> dataWeights_;

public:
  CorpusBatch(const std::vector<Ptr<SubBatch>>& batches) : batches_(batches) {}

  /**
   * @brief Access i-th subbatch storing a source or target sentence.
   *
   * The order of subbatches is: 1st source sentence, 2nd source sentence, ...,
   * target sentence.
   *
   * @param i position of the element to return
   *
   * @return Pointer to the requested element.
   */
  Ptr<SubBatch> operator[](size_t i) const { return batches_[i]; }

  /**
   * @brief Access the first subbatch, i.e. the source sentence.
   */
  Ptr<SubBatch> front() { return batches_.front(); }

  /**
   * @brief Access the last subbatch, i.e. the target sentence.
   */
  Ptr<SubBatch> back() { return batches_.back(); }

  /**
   * @brief The number of sentences in the batch.
   */
  size_t size() const { return batches_[0]->batchSize(); }

  /**
   * @brief The total number of words for the longest sentence in the batch plus one. Pass which=0 for source and -1 for target.
   */
  size_t words(int which = 0) const { return batches_[which >= 0 ? which : which + (ptrdiff_t)batches_.size()]->batchWords(); }

  /**
   * @brief The width of the source mini-batch. Num words + padded?
   */
  size_t width() const { return batches_[0]->batchWidth(); }

  /**
   * @brief The number of sentences in the batch, target words.
   */
  size_t sizeTrg() const { return batches_.back()->batchSize(); }
  
  /**
   * @brief The number of words for the longest sentence in the batch plus one.
   */
  size_t wordsTrg() const { return batches_.back()->batchWords(); };

  /**
   * @brief The width of the target mini-batch. Num words + padded?
   */
  size_t widthTrg() const { return batches_.back()->batchWidth(); };

  /**
   * @brief The number of source and targets.
   */
  size_t sets() const { return batches_.size(); }

  /**
   * @brief Creates a batch filled with fake data. Used to determine the size of
   * the batch object.
   *
   * @param lengths List of subbatch sizes.
   * @param batchSize Number of sentences in the batch.
   * @param options Options with "guided-alignment" and "data-weighting".
   *
   * @return Fake batch of the same size as the real batch.
   */
  static Ptr<CorpusBatch> fakeBatch(std::vector<size_t>& lengths,
                                    size_t batchSize,
                                    Ptr<Options> options) {
    std::vector<Ptr<SubBatch>> batches;

    for(auto len : lengths) {
      auto sb = New<SubBatch>(batchSize, len);
      std::fill(sb->mask().begin(), sb->mask().end(), 1);

      batches.push_back(sb);
    }

    auto batch = New<CorpusBatch>(batches);
    
    if(!options) return batch;

    if(options->has("guided-alignment")) {
      std::vector<float> alignment(batchSize * lengths.front() * lengths.back(),
                                   0.f);
      batch->setGuidedAlignment(alignment);
    }

    if(options->has("data-weighting")) {
      int weightsSize = batchSize;
      if(options->get<std::string>("data-weighting-type") != "sentence")
        weightsSize *= lengths.back();
      std::vector<float> weights(weightsSize, 1.f);
      batch->setDataWeights(weights);
    }

    return batch;
  }

  /**
   * @brief Splits the batch into batches of equal size.
   *
   * @param n number of splits
   *
   * @return Vector of pointers to new batches.
   *
   * @see marian::data::SubBatch::split(size_t n)
   */
  std::vector<Ptr<Batch>> split(size_t n) {
    // split each subbatch separately
    std::vector<std::vector<Ptr<SubBatch>>> subs(n);
    for(auto subBatch : batches_) {
      size_t i = 0;
      for(auto splitSubBatch : subBatch->split(n))
        subs[i++].push_back(splitSubBatch);
    }

    // create batches from split subbatches
    std::vector<Ptr<Batch>> splits;
    for(auto subBatches : subs)
      splits.push_back(New<CorpusBatch>(subBatches));

    // set sentence indices in split batches
    size_t pos = 0;
    for(auto split : splits) {
      std::vector<size_t> ids;
      for(int i = pos; i < pos + split->size(); ++i)
        ids.push_back(sentenceIds_[i]);
      split->setSentenceIds(ids);
      pos += split->size();
    }

    // @TODO: restore word alignments in split batches
    ABORT_IF(
        !guidedAlignment_.empty(),
        "Guided alignment with synchronous SGD is temporarily not supported");

    // restore data weights in split batches
    pos = 0;
    if(!dataWeights_.empty()) {
      size_t oldSize = size();

      size_t width = 1;
      // There are more weights than sentences, i.e. these are word weights.
      if(dataWeights_.size() != oldSize)
        width = batches_.back()->batchWidth();

      for(auto split : splits) {
        std::vector<float> ws(width * split->size(), 1.0f);

        // this needs to be split along the batch dimension
        // which is here the innermost dimension.
        // Should work for sentence-based weights, too.
        for(int j = 0; j < width; ++j) {
          for(int i = 0; i < split->size(); ++i) {
            ws[j * split->size() + i] = dataWeights_[j * oldSize + i + pos];
          }
        }
        split->setDataWeights(ws);
        pos += split->size();
      }
    }

    return splits;
  }

  std::vector<float>& getGuidedAlignment() { return guidedAlignment_; }
  void setGuidedAlignment(const std::vector<float>& aln) {
    guidedAlignment_ = aln;
  }

  std::vector<float>& getDataWeights() { return dataWeights_; }
  void setDataWeights(const std::vector<float>& weights) {
    dataWeights_ = weights;
  }

  /**
   * @brief Prints the batch in a readable form on stderr for debugging.
   */
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
          size_t idx = i * sb->batchSize() + j;
          Word w = sb->data()[idx];
          std::cerr << w << " ";
        }
        std::cerr << std::endl;
      }
    }

    if(!dataWeights_.empty()) {
      std::cerr << "weights: ";
      for(auto w : dataWeights_)
        std::cerr << w << " ";
      std::cerr << std::endl;
    }
  }
};

class CorpusIterator;

class CorpusBase
    : public DatasetBase<SentenceTuple, CorpusIterator, CorpusBatch>,
      public RNGEngine {
public:
  CorpusBase() : DatasetBase() {}

  CorpusBase(Ptr<Config> options, bool translate = false);

  CorpusBase(std::vector<std::string> paths,
             std::vector<Ptr<Vocab>> vocabs,
             Ptr<Config> options);

  virtual std::vector<Ptr<Vocab>>& getVocabs() = 0;

protected:
  std::vector<UPtr<InputFileStream>> files_;
  std::vector<Ptr<Vocab>> vocabs_;

  size_t pos_{0};

  Ptr<Config> options_;

  size_t maxLength_{0};
  bool maxLengthCrop_{false};
  bool rightLeft_{false};

  /**
   * @brief Index of the file with weights in paths_ and files_; zero means no
   * weights file provided.
   */
  size_t weightFileIdx_{0};

  /**
   * @brief Index of the file with alignments in paths_ and files_; zero means
   * no alignment file provided.
   */
  size_t alignFileIdx_{0};

  /**
   * @brief Helper function converting a line of text into words using the i-th
   * vocabulary and adding them to the sentence tuple.
   */
  void addWordsToSentenceTuple(const std::string& line,
                               size_t i,
                               SentenceTuple& tup) const;
  /**
   * @brief Helper function parsing a line with word alignments and adding them
   * to the sentence tuple.
   */
  void addAlignmentToSentenceTuple(const std::string& line,
                                   SentenceTuple& tup) const;
  /**
   * @brief Helper function parsing a line of weights and adding them to the
   * sentence tuple.
   */
  void addWeightsToSentenceTuple(const std::string& line,
                                 SentenceTuple& tup) const;

  void addAlignmentsToBatch(Ptr<CorpusBatch> batch,
                            const std::vector<sample>& batchVector);

  void addWeightsToBatch(Ptr<CorpusBatch> batch,
                         const std::vector<sample>& batchVector);
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
}
}
