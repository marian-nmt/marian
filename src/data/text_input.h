#pragma once

#include <boost/iterator/iterator_facade.hpp>

#include "data/corpus.h"

namespace marian {
namespace data {

class TextInput;

class TextIterator
    : public boost::iterator_facade<TextIterator,
                                    SentenceTuple const,
                                    boost::forward_traversal_tag> {
public:
  TextIterator();
  explicit TextIterator(TextInput& corpus);

private:
  friend class boost::iterator_core_access;

  void increment();

  bool equal(TextIterator const& other) const;

  const SentenceTuple& dereference() const;

  TextInput* corpus_;

  long long int pos_;
  SentenceTuple tup_;
};

class TextInput : public DatasetBase<SentenceTuple, TextIterator, CorpusBatch> {
private:
  Ptr<Config> options_;

  std::vector<UPtr<std::istringstream>> files_;
  std::vector<Ptr<Vocab>> vocabs_;

  size_t pos_{0};

public:
  TextInput(std::vector<std::string> inputs,
            std::vector<Ptr<Vocab>> vocabs,
            Ptr<Config> options);

  sample next();

  void shuffle() {}
  void reset() {}

  iterator begin() { return iterator(*this); }
  iterator end() { return iterator(); }

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
          subBatches[j]->data()[k * batchSize + i] = batchVector[i][j][k];
          subBatches[j]->mask()[k * batchSize + i] = 1.f;
          words[j]++;
        }
      }
    }

    for(size_t j = 0; j < maxDims.size(); ++j)
      subBatches[j]->setWords(words[j]);

    auto batch = batch_ptr(new batch_type(subBatches));
    batch->setSentenceIds(sentenceIds);

    return batch;
  }

  void prepare() {}
};
}
}
