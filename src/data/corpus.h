#pragma once

#include <iostream>
#include <fstream>
#include <boost/iterator/iterator_facade.hpp>

#include "common/definitions.h"
#include "data/vocab.h"
#include "common/file_stream.h"

namespace marian {
namespace data {

typedef std::vector<size_t> WordBatch;
typedef std::vector<float> MaskBatch;
typedef std::pair<WordBatch, MaskBatch> WordMask;
typedef std::vector<WordMask> SentBatch;

typedef std::vector<Words> SentenceTuple;

class CorpusBatch {
  public:
    CorpusBatch(const std::vector<SentBatch>& batches, size_t words = 0)
    : batches_(batches), words_(words) {}

    const SentBatch& operator[](size_t i) const {
      return batches_[i];
    }

    void debug() {
      size_t i = 0;
      for(auto l : batches_) {
        std::cerr << "input " << i++ << ": " << std::endl;
        for(auto b : l) {
          std::cerr << "\t w: ";
          for(auto w : b.first) {
            std::cerr << w << " ";
          }
          std::cerr << std::endl;

          std::cerr << "\t m: ";
          for(auto w : b.second) {
            std::cerr << w << " ";
          }
          std::cerr << std::endl;
        }
      }
    }

    size_t size() const {
      return batches_[0][0].first.size();
    }

    size_t words() const {
      return words_;
    }

  private:
    std::vector<SentBatch> batches_;
    size_t words_;
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
    std::vector<std::string> textPaths_;
    std::vector<UPtr<InputFileStream>> files_;
    std::vector<Vocab> vocabs_;
    size_t maxLength_;

    void shuffleFiles(const std::vector<std::string>& paths);

  public:
    typedef CorpusBatch batch_type;
    typedef Ptr<batch_type> batch_ptr;

    typedef CorpusIterator iterator;
    typedef SentenceTuple sample;

    Corpus(const std::vector<std::string>& textPaths,
           const std::vector<std::string>& vocabPaths,
           const std::vector<int>& maxVocabs,
           size_t maxLength = 50);

    sample next();

    void shuffle();

    iterator begin() {
      return iterator(*this);
    }

    iterator end() {
      return iterator();
    }

    batch_ptr toBatch(const std::vector<sample>& batchVector) {
      int batchSize = batchVector.size();
      size_t words = 0;

      std::vector<int> maxDims;
      for(auto& ex : batchVector) {
        if(maxDims.size() < ex.size())
          maxDims.resize(ex.size(), 0);
        for(int i = 0; i < ex.size(); ++i) {
          if(ex[i].size() > maxDims[i])
          maxDims[i] = ex[i].size();
        }
      }

      std::vector<SentBatch> langs;
      for(auto m : maxDims) {
        langs.push_back(SentBatch(m,
                                  { WordBatch(batchSize, 0),
                                    MaskBatch(batchSize, 0) } ));
      }

      for(int i = 0; i < batchSize; ++i) {
        for(int j = 0; j < maxDims.size(); ++j) {
          for(int k = 0; k < batchVector[i][j].size(); ++k) {
            langs[j][k].first[i] = batchVector[i][j][k];
            langs[j][k].second[i] = 1.f;
            if(j == 0)
              words++;
          }
        }
      }

      //words -= batchSize; // subtract end of sentence token

      return batch_ptr(new batch_type(langs, words));
    }
};

}
}
