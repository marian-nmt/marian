#pragma once

#include "common/definitions.h"
#include "common/file_stream.h"
#include "common/options.h"
#include "common/utils.h"
#include "data/alignment.h"
#include "data/iterator_facade.h"
#include "data/batch.h"
#include "data/dataset.h"
#include "data/rng_engine.h"
#include "data/vocab.h"

#include <future>

namespace marian {
namespace data {

/**
 * @brief A sentence tuple that stores all sources and target sentences for a
 * specific "line" from a parallel corpus.
 *
 * Sentence tuples store sentences from external files and a basis for
 * construction of marian::data::CorpusBatch objects. They are not a part of
 * marian::data::CorpusBatch.
 */
class SentenceTupleImpl {
private:
  size_t id_;
  std::vector<Words> tuple_;    // [stream index][step index]
  std::vector<float> weights_;  // [stream index]
  WordAlignment alignment_;
  bool altered_ = false;

public:
  typedef Words value_type;

  /**
   * @brief Creates an empty tuple with 0 id (default constructor).
   */
  SentenceTupleImpl() : id_(0) {}

  /**
   * @brief Creates an empty tuple with the given Id.
   */
  SentenceTupleImpl(size_t id) : id_(id) {}

  ~SentenceTupleImpl() {}

  /**
   * @brief Returns the sentence's ID.
   */
  size_t getId() const { return id_; }

  /**
   * @brief Returns whether this Tuple was altered or augmented from what
   * was provided to Marian in input.
   */
  bool isAltered() const { return altered_; }

  /**
   * @brief Mark that this Tuple was internally altered or augmented by Marian
   */
  void markAltered() { altered_ = true; }

  /**
   * @brief Adds a new sentence at the end of the tuple.
   *
   * @param words A vector of word indices.
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
   * @brief Get sentence weights.
   *
   * For sentence-level weights the vector contains only one element.
   */
  const std::vector<float>& getWeights() const { return weights_; }

  /**
   * @brief Set sentence weights.
   *
   * For word weights (i.e. weights.size() > 1) it checks if there is as many weights as
   * target tokens. If not, it aborts. Because of this, this function must becalled *after*
   * adding source and target tokens.
   */
  void setWeights(const std::vector<float>& weights);

  const WordAlignment& getAlignment() const { return alignment_; }
  void setAlignment(const WordAlignment& alignment) { alignment_ = alignment; }
};

class SentenceTuple {
private:
  std::shared_ptr<std::future<SentenceTupleImpl>> fImpl_;
  mutable std::shared_ptr<SentenceTupleImpl> impl_;

public:
  typedef Words value_type;

  /**
   * @brief Creates an empty tuple with no associated future.
   */
  SentenceTuple() {}
  
  SentenceTuple(const SentenceTupleImpl& tupImpl) 
    : impl_(std::make_shared<SentenceTupleImpl>(tupImpl)) {}

  SentenceTuple(std::future<SentenceTupleImpl>&& fImpl) 
    : fImpl_(new std::future<SentenceTupleImpl>(std::move(fImpl))) {}

  SentenceTupleImpl& get() const {
    if(!impl_) {
      ABORT_IF(!fImpl_ || !fImpl_->valid(), "No future tuple associated with SentenceTuple");
      impl_ = std::make_shared<SentenceTupleImpl>(fImpl_->get());
    }
    return *impl_;
  }

  /**
   * @brief Returns the sentence's ID.
   */
  size_t getId() const { return get().getId(); }

  /**
   * @brief Returns whether this Tuple was altered or augmented from what
   * was provided to Marian in input.
   */
  bool isAltered() const { return get().isAltered(); }

  /**
   * @brief The size of the tuple, e.g. two for parallel data with a source and
   * target sentences.
   */
  size_t size() const { return get().size(); }

  /**
   * @brief confirms that the tuple has been populated with data
   */
  bool valid() const {
    return fImpl_ || impl_;
  }

  /**
   * @brief The i-th tuple sentence.
   *
   * @param i Tuple's index.
   */
  Words& operator[](size_t i) { return get()[i]; }
  const Words& operator[](size_t i) const { return get()[i]; }

  /**
   * @brief The last tuple sentence, i.e. the target sentence.
   */
  Words& back() { return get().back(); }
  const Words& back() const { return get().back(); }

  /**
   * @brief Checks whether the tuple is empty.
   */
  bool empty() const { return get().empty(); }

  auto begin() const -> decltype(get().begin()) { return get().begin(); }
  auto end() const -> decltype(get().end()) { return get().end(); }

  auto rbegin() const -> decltype(get().rbegin()) { return get().rbegin(); }
  auto rend() const -> decltype(get().rend()) { return get().rend(); }

  /**
   * @brief Get sentence weights.
   *
   * For sentence-level weights the vector contains only one element.
   */
  const std::vector<float>& getWeights() const { return get().getWeights(); }

  const WordAlignment& getAlignment() const { return get().getAlignment(); }
};

/**
 * @brief Batch of sentences represented as word indices with masking.
 */
class SubBatch {
private:
  Words indices_;
  std::vector<float> mask_;

  size_t size_;
  size_t width_;
  size_t words_;

  Ptr<const Vocab> vocab_;
  // ... TODO: add the length information (remember it)

public:
  /**
   * @brief Creates an empty subbatch of specified size.
   *
   * @param size Number of sentences
   * @param width Number of words in the longest sentence
   */
  SubBatch(size_t size, size_t width, const Ptr<const Vocab>& vocab)
    : indices_(size * width, vocab ? vocab->getEosId() : Word::ZERO), // note: for gaps, we must use a valid index
    mask_(size * width, 0),
    size_(size),
    width_(width),
    words_(0),
    vocab_(vocab) {}

  /**
   * @brief Flat vector of word indices.
   *
   * The order of indices is \f$idx_{0,0}, idx_{0,1},\dots,idx_{0,s}, \dots,
   * idx_{w,0},idx_{w,1},\dots,idx_{w,s}\f$, where \f$w\f$ is the number of
   * words (width) and \f$s\f$ is the number of sentences (size).
   */
  Words& data() { return indices_; }
  const Words& data() const { return indices_; }
  /**
   * @brief compute flat index into data() and mask() vectors for given batch index and word index in sentence
   */
  size_t locate(size_t batchIdx, size_t wordPos) const { return locate(batchIdx, wordPos, size_); }
  static size_t locate(size_t batchIdx, size_t wordPos, size_t batchSize) { return wordPos * batchSize + batchIdx; }
  /**
   * @brief Flat masking vector; 0 is used for masked words.
   *
   * @see data()
   */
  std::vector<float>& mask() { return mask_; }
  const std::vector<float>& mask() const { return mask_; }

  /**
   * @brief Accessors to the vocab_ field.
   */
  const Ptr<const Vocab>& vocab() const { return vocab_; }

  /**
   * @brief The number of sentences in the batch.
   */
  size_t batchSize() const { return size_; }
  /**
   * @brief The number of words in the longest sentence in the batch.
   */
  size_t batchWidth() const { return width_; };
  /**
   * @brief The total number of words in the batch (not counting masked-out words).
   */
  size_t batchWords() const { return words_; }

  /**
   * @brief Splits the stream into sub-batches of equal size (except for last).
   *
   * @param n number of sub-batches to split into
   *
   * @param sizeLimit Pretend the batch only has this many sentences. Used for MB-size ramp-up.
   *
   * @return Vector of pointers to new sub-batches (or nullptrs where run out of sub-batches)
   *
   * @see marian::data::Batch::split(size_t n)
   */
  std::vector<Ptr<SubBatch>> split(size_t n, size_t sizeLimit /*or SIZE_MAX*/) const {
    ABORT_IF(size_ == 0, "Encountered sub-batch size of 0");

    auto size = std::min(size_, sizeLimit); // if limit is given then pretend the batch only has that many sentences
    size_t targetSubSize = (size_t)(std::ceil(size / (float)n)); // aim at forming sub-batches of this #sentences

    std::vector<Ptr<SubBatch>> splits;
    for(size_t pos = 0; pos < size; pos += targetSubSize) { // loop over ranges of size targetSubSize to form sub-batches of this size
      size_t subSize = std::min(targetSubSize, size - pos); // actual number of sentences can be smaller at the end

      // determine actual width (=max length) of this sub-batch, which may be smaller than the overall max length
      size_t subWidth = 0;
      for(size_t s = 0; s < width_; ++s) {
        for(size_t b = 0; b < subSize; ++b) {
          if(mask_[locate(/*batchIdx=*/pos + b, /*wordPos=*/s)] != 0)   // s * size_ + (pos + b)
            if (subWidth < s + 1)
              subWidth = s + 1;
        }
      }

      // create sub-batch
      auto sb = New<SubBatch>(subSize, subWidth, vocab_);

      size_t words = 0;
      for(size_t s = 0; s < subWidth; ++s) {
        for(size_t b = 0; b < subSize; ++b) {
          sb->data()[locate(/*batchIdx=*/b, /*wordPos=*/s, /*batchSize=*/subSize)/*s * subSize + b*/] = indices_[locate(/*batchIdx=*/pos + b, /*wordPos=*/s)]; // s * size_ + (pos + b)
          sb->mask()[locate(/*batchIdx=*/b, /*wordPos=*/s, /*batchSize=*/subSize)/*s * subSize + b*/] =    mask_[locate(/*batchIdx=*/pos + b, /*wordPos=*/s)]; // s * size_ + (pos + b)

          if(mask_[locate(/*batchIdx=*/pos + b, /*wordPos=*/s)/*s * size_ + (pos + b)*/] != 0)
            words++;
        }
      }
      sb->setWords(words);

      splits.push_back(sb);
    }
    return splits;
  }

  void setWords(size_t words) { words_ = words; }
};

/**
 * @brief Batch of source(s) and target sentences with additional information,
 * such as guided alignments and sentence or word-level weighting.
 */
class CorpusBatch : public Batch {
protected:
  std::vector<Ptr<SubBatch>> subBatches_;
  std::vector<float> guidedAlignment_; // [max source len, batch size, max target len] flattened
  std::vector<float> dataWeights_;

public:
  CorpusBatch(const std::vector<Ptr<SubBatch>>& subBatches)
    : subBatches_(subBatches) {}

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
  Ptr<SubBatch> operator[](size_t i) const { return subBatches_[i]; }

  /**
   * @brief Access the first subbatch, i.e. the source sentence.
   */
  Ptr<SubBatch> front() { return subBatches_.front(); }

  /**
   * @brief Access the last subbatch, i.e. the target sentence.
   */
  Ptr<SubBatch> back() { return subBatches_.back(); }

  /**
   * @brief The number of sentences in the batch.
   */
  size_t size() const override { return subBatches_[0]->batchSize(); }

  /**
   * @brief The total number of words in the batch (not counting masked-out words).
   * Pass which=0 for source words and -1 for target words.
   */
  size_t words(int which = 0) const override {
    return subBatches_[which >= 0 ? which
      : which + (ptrdiff_t)subBatches_.size()]
      ->batchWords();
  }

  /**
   * @brief The width of the source mini-batch. Num words + padded?
   */
  size_t width() const override { return subBatches_[0]->batchWidth(); }

  /**
   * @brief The number of sentences in the batch, target words.
   */
  size_t sizeTrg() const override { return subBatches_.back()->batchSize(); }

  /**
   * @brief The total number of words in the batch (not counting masked-out words).
   */
  size_t wordsTrg() const override { return subBatches_.back()->batchWords(); };

  /**
   * @brief The target width (=max length) of the mini-batch.
   */
  size_t widthTrg() const override { return subBatches_.back()->batchWidth(); };

  /**
   * @brief The number of source and targets.
   */
  size_t sets() const { return subBatches_.size(); }

  /**
   * @brief Creates a batch filled with fake data. Used to determine the size of
   * the batch object. With guided-alignments and multiple encoders, those
   * multiple source streams are expected to have the same lengths.
   *
   * @param lengths List of subbatch sizes.
   * @param batchSize Number of sentences in the batch.
   * @param options Options with "guided-alignment" and "data-weighting".
   *
   * @return Fake batch of the same size as the real batch.
   */
  static Ptr<CorpusBatch> fakeBatch(const std::vector<size_t>& lengths,
      const std::vector<Ptr<Vocab>>& vocabs,
      size_t batchSize,
      Ptr<Options> options) {
    std::vector<Ptr<SubBatch>> batches;

    size_t batchIndex = 0;
    for(auto len : lengths) {
      auto sb = New<SubBatch>(batchSize, len, vocabs[batchIndex]);
      // set word indices to random values (not actually needed with current version  --@marcinjd: please confirm)
      std::transform(sb->data().begin(), sb->data().end(), sb->data().begin(),
          [&](Word) -> Word { return vocabs[batchIndex]->randWord(); });
      // mask: no items ask being masked out
      std::fill(sb->mask().begin(), sb->mask().end(), 1.f);
      batchIndex++;

      batches.push_back(sb);
    }

    auto batch = New<CorpusBatch>(batches);

    if(!options)
      return batch;

    if(options->get("guided-alignment", std::string("none")) != "none") {
      // @TODO: if > 1 encoder, verify that all encoders have the same sentence lengths
      std::vector<float> alignment(batchSize * lengths.front() * lengths.back(),
          0.f);
      batch->setGuidedAlignment(std::move(alignment));
    }

    if(options->hasAndNotEmpty("data-weighting")) {
      auto weightsSize = batchSize;
      if(options->get<std::string>("data-weighting-type") != "sentence")
        weightsSize *= lengths.back();
      std::vector<float> weights(weightsSize, 1.f);
      batch->setDataWeights(weights);
    }

    return batch;
  }

  /**
   * @brief Splits the batch into batches of equal size (except for last).
   *
   * @param n number of sub-batches to split into
   *
   * @param sizeLimit Clip batch content to the first sizeLimit sentences in the batch
   *
   * @return Vector of pointers to new sub-batches (or nullptrs where run out of sub-batches)
   *
   * @see marian::data::SubBatch::split(size_t n)
   */
  std::vector<Ptr<Batch>> split(size_t n, size_t sizeLimit /*=SIZE_MAX*/) override {
    ABORT_IF(size() == 0, "Encountered batch size of 0");

    std::vector<std::vector<Ptr<SubBatch>>> subs; // [subBatchIndex][streamIndex]
    // split each stream separately
    for(auto batchStream : subBatches_) {
      size_t i = 0; // index into split batch
      for(auto splitSubBatch : batchStream->split(n, sizeLimit)) { // splits a batch into pieces, can also change width
        if(subs.size() <= i)
          subs.resize(i + 1);
        subs[i++].push_back(splitSubBatch); // this forms tuples across streams
      }
    }

    // create batches from split subbatches
    std::vector<Ptr<Batch>> splits;
    for(auto subBatches : subs)
      splits.push_back(New<CorpusBatch>(subBatches));

    // set sentence indices in split batches
    size_t pos = 0;
    for(auto split : splits) {
      std::vector<size_t> ids;
      for(size_t i = pos; i < pos + split->size(); ++i)
        ids.push_back(sentenceIds_[i]);
      split->setSentenceIds(ids);
      pos += split->size();
    }

    if(!guidedAlignment_.empty()) {
      size_t oldTrgWords = back()->batchWidth();
      size_t oldSize = size();

      pos = 0;
      for(auto split : splits) {
        auto cb = std::static_pointer_cast<CorpusBatch>(split);
        size_t srcWords = cb->front()->batchWidth();
        size_t trgWords = cb->back()->batchWidth();
        size_t dimBatch = cb->size();

        std::vector<float> aligns(srcWords * dimBatch * trgWords, 0.f);

        for(size_t i = 0; i < dimBatch; ++i) {
          size_t bi = i + pos;
          for(size_t sid = 0; sid < srcWords; ++sid) {
            for(size_t tid = 0; tid < trgWords; ++tid) {
              size_t bidx = sid * oldSize  * oldTrgWords + bi * oldTrgWords + tid; // [sid, bi, tid]
              size_t idx  = sid * dimBatch *    trgWords +  i *    trgWords + tid;
              aligns[idx] = guidedAlignment_[bidx];
            }
          }
        }
        cb->setGuidedAlignment(std::move(aligns));
        pos += dimBatch;
      }
    }

    // restore data weights in split batches
    pos = 0;
    if(!dataWeights_.empty()) {
      size_t oldSize = size();

      for(auto split : splits) {
        auto cb = std::static_pointer_cast<CorpusBatch>(split);
        size_t width = 1;                   // One weight per sentence in case of sentence-level weights
        if(dataWeights_.size() != oldSize)  // if number of weights does not correspond to number of sentences we have word-level weights
          width = cb->back()->batchWidth(); // splitting also affects width, hence we need to accomodate this here
        std::vector<float> ws(width * split->size(), 1.0f);

        // this needs to be split along the batch dimension
        // which is here the innermost dimension.
        // Should work for sentence-based weights, too.
        for(size_t s = 0; s < width; ++s) {
          for(size_t b = 0; b < split->size(); ++b) {
            ws[s * split->size() + b] = dataWeights_[s * oldSize + b + pos]; // @TODO: use locate() as well
          }
        }
        split->setDataWeights(ws);
        pos += split->size();
      }
    }

    return splits;
  }

  const std::vector<float>& getGuidedAlignment() const { return guidedAlignment_; }  // [dimSrcWords, dimBatch, dimTrgWords] flattened
  void setGuidedAlignment(std::vector<float>&& aln) override {
    guidedAlignment_ = std::move(aln);
  }

  size_t locateInGuidedAlignments(size_t b, size_t s, size_t t) {
    return ((s * size()) + b) * widthTrg() + t;
  }

  std::vector<float>& getDataWeights() { return dataWeights_; }
  void setDataWeights(const std::vector<float>& weights) override {
    dataWeights_ = weights;
  }

  /**
   * @brief Prints the batch in a readable form on stderr for debugging.
   */
  void debug(bool printIndices = false) override { // prints word string if subbatch has vocab and
    // printIndices == false otherwise only numeric indices
    std::cerr << "batches: " << sets() << std::endl;

    if(!sentenceIds_.empty()) {
      std::cerr << "indices: ";
      for(auto id : sentenceIds_)
        std::cerr << id << " ";
      std::cerr << std::endl;
    }

    size_t subBatchIndex = 0;
    for(auto sb : subBatches_) {
      std::cerr << "stream " << subBatchIndex++ << ": " << std::endl;
      const auto& vocab = sb->vocab();
      for(size_t s = 0; s < sb->batchWidth(); s++) {
        std::cerr << "\t w: ";
        for(size_t b = 0; b < sb->batchSize(); b++) {
          Word w = sb->data()[sb->locate(/*batchIdx=*/b, /*wordPos=*/s)]; // s * sb->batchSize() + b;
          if (vocab && !printIndices)
            std::cerr << (*vocab)[w] << " ";
          else
            std::cerr << w.toString() << " "; // if not loaded then print numeric id instead
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

class CorpusBase : public DatasetBase<SentenceTuple, CorpusIterator, CorpusBatch>, public RNGEngine {
public:
  typedef SentenceTuple Sample;

  CorpusBase(Ptr<Options> options,
             bool translate = false,
             size_t seed = Config::seed);

  CorpusBase(const std::vector<std::string>& paths,
             const std::vector<Ptr<Vocab>>& vocabs,
             Ptr<Options> options,
             size_t seed = Config::seed);

  virtual ~CorpusBase() {}
  virtual std::vector<Ptr<Vocab>>& getVocabs() = 0;

protected:
  std::vector<UPtr<std::istream>> files_;
  std::vector<Ptr<Vocab>> vocabs_;

  /**
   * @brief Determines if a EOS symbol should be added. By default this is true for any sequence,
   * but should be false for instance for classifier labels. This is set per input stream, hence a
   * vector.
   */
  std::vector<bool> addEOS_;

  size_t pos_{0};

  size_t maxLength_{0};
  bool maxLengthCrop_{false};
  bool rightLeft_{false};

  bool tsv_{false};  // true if the input is a single file with tab-separated values
  size_t tsvNumInputFields_{0};  // number of fields from the TSV input that are associated
                                  // with vocabs, i.e. excluding fields with alignment or
                                  // weights, only if --tsv
  /**
   * @brief Determine the number of fields from the TSV input that are associated with
   * vocabs, i.e. excluding fields that contain alignment or weights
   */
  static size_t getNumberOfTSVInputFields(Ptr<Options> options);

  /**
   * @brief Index of the file with weights in paths_ and files_; -1 means no
   * weights file provided.
   */
  int weightFileIdx_{-1};

  /**
   * @brief Index of the file with alignments in paths_ and files_; -1 means
   * no alignment file provided.
   */
  int alignFileIdx_{-1};

  /**
   * @brief Determine if EOS symbol should be added to input
   */
  void initEOS(bool training);

  /**
   * @brief Helper function converting a line of text into words using the i-th
   * vocabulary and adding them to the sentence tuple.
   */
  void addWordsToSentenceTuple(const std::string& line, size_t batchIndex, SentenceTupleImpl& tup) const;
  /**
   * @brief Helper function parsing a line with word alignments and adding them
   * to the sentence tuple.
   */
  void addAlignmentToSentenceTuple(const std::string& line, SentenceTupleImpl& tup) const;
  /**
   * @brief Helper function parsing a line of weights and adding them to the
   * sentence tuple.
   */
  void addWeightsToSentenceTuple(const std::string& line, SentenceTupleImpl& tup) const;

  void addAlignmentsToBatch(Ptr<CorpusBatch> batch, const std::vector<Sample>& batchVector);

  void addWeightsToBatch(Ptr<CorpusBatch> batch, const std::vector<Sample>& batchVector);
};

class CorpusIterator : public IteratorFacade<CorpusIterator, SentenceTuple> {
public:
  CorpusIterator();
  explicit CorpusIterator(CorpusBase* corpus);

private:
  void increment() override;

  bool equal(CorpusIterator const& other) const override;

  const SentenceTuple& dereference() const override;

  CorpusBase* corpus_;

  int64_t pos_; // we use int64_t here because the initial value can be -1
  SentenceTuple tup_;
};
}  // namespace data
}  // namespace marian
