#pragma once

#include <iostream>
#include <fstream>
#include <random>
#include <boost/iterator/iterator_facade.hpp>

#include "training/config.h"
#include "common/definitions.h"
#include "data/vocab.h"
#include "common/file_stream.h"

namespace marian {
namespace data {

typedef std::vector<Words> SentenceTuple;

class SubBatch {
  private:
    std::vector<Word> indeces_;
    std::vector<float> mask_;
    
    int size_;
    int width_;
    int words_;
    
  public:
    SubBatch(const std::vector<Word> indeces,
             const std::vector<float> mask,
             int size, int width, int words)
    : indeces_(indeces), mask_(mask),
      size_(size), width_(width), words_(words)
    { }
    
    std::vector<Word>& indeces() { return indeces_; }
    std::vector<float>& mask()   { return mask_; }
    
    int batchSize()   { return size_; }
    int batchWidth() { return width_; };
    int batchWords()  { return words_; }
    
};

class CorpusBatch {
  public:
    CorpusBatch(const std::vector<Ptr<SubBatch>>& batches)
    : batches_(batches) {}

    Ptr<SubBatch> operator[](size_t i) const {
      return batches_[i];
    }
    
    Ptr<SubBatch> back() {
      return batches_.back();
    }

    void debug() {
      size_t i = 0;
      for(auto sb : batches_) {
        std::cerr << "input " << i++ << ": " << std::endl;
        for(size_t i = 0; i < sb->batchSize(); i++) {
          std::cerr << "\t w: ";
          for(size_t j = 0; j < sb->batchWidth(); j++) {
            Word w = sb->indeces()[i * sb->batchWidth() + j];
            std::cerr << w << " ";
          }
          std::cerr << std::endl;
        }
      }
    }

    size_t size() const {
      return batches_[0]->batchSize();
    }

    size_t words() const {
      return batches_[0]->batchWords();
    }

    size_t sets() const {
      return batches_.size();
    }
    
    static Ptr<CorpusBatch> fakeBatch(std::vector<size_t>& lengths, size_t batchSize) {
      std::vector<Ptr<SubBatch>> batches;
      
      for(auto len : lengths) {
        std::vector<Word> indeces(batchSize * len, 0);
        std::vector<float> mask(batchSize * len, 0);
        auto sb = New<SubBatch>(indeces, mask, batchSize, len, batchSize * len);
        batches.push_back(sb);
      }
        
      return New<CorpusBatch>(batches);
    }

  private:
    std::vector<Ptr<SubBatch>> batches_;
};

class Corpus;

class CorpusIterator
  : public boost::iterator_facade<CorpusIterator,
                                  SentenceTuple const,
                                  boost::forward_traversal_tag>
{
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

class Corpus {
  private:
    Ptr<Config> options_;

    std::vector<std::string> textPaths_;
    std::vector<UPtr<TemporaryFile>> tempFiles_;
    std::vector<UPtr<InputFileStream>> files_;
    std::vector<Ptr<Vocab>> vocabs_;
    size_t maxLength_;

    std::random_device rd_;
    std::mt19937 g_;

    void shuffleFiles(const std::vector<std::string>& paths);

  public:
    typedef CorpusBatch batch_type;
    typedef Ptr<batch_type> batch_ptr;

    typedef CorpusIterator iterator;
    typedef SentenceTuple sample;

    Corpus(Ptr<Config> options, bool translate=false);

    Corpus(std::vector<std::string> paths,
           std::vector<Ptr<Vocab>> vocabs,
           Ptr<Config> options,
           size_t maxLength = 0);

    sample next();

    void shuffle();

    void reset();

    iterator begin() {
      return iterator(*this);
    }

    iterator end() {
      return iterator();
    }

    std::vector<Ptr<Vocab>>& getVocabs() {
      return vocabs_;
    }

    batch_ptr toBatch(const std::vector<sample>& batchVector) {
      int batchSize = batchVector.size();

      std::vector<int> maxDims;
      for(auto& ex : batchVector) {
        if(maxDims.size() < ex.size())
          maxDims.resize(ex.size(), 0);
        for(int i = 0; i < ex.size(); ++i) {
          if(ex[i].size() > maxDims[i])
          maxDims[i] = ex[i].size();
        }
      }

      std::vector<Ptr<SubBatch>> subBatches;
      size_t i = 0;
      for(auto width : maxDims) {
        std::vector<Word> indeces(batchSize * width, 0);
        std::vector<float> mask(batchSize * width, 0);
        
        int words = 0;
        auto itInd = indeces.begin();
        auto itMsk = mask.begin();
        for(auto& sample : batchVector) {
          auto& line = sample[i];
          std::copy(line.begin(), line.end(), itInd);
          std::fill(itMsk, itMsk + width, 1.f);
          words += line.size();
          
          itInd += width;
          itMsk += width;
        }
        subBatches.push_back(New<SubBatch>(indeces, mask, batchSize, width, words));
        i++;
      }
      
      return batch_ptr(new batch_type(subBatches));
    }
};

}
}
