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
#include "data/corpus.h"

#include <SQLiteCpp/SQLiteCpp.h>

namespace marian {
namespace data {

class CorpusSQLite : public CorpusBase {
private:
  Ptr<Config> options_;

  std::vector<UPtr<InputFileStream>> files_;
  std::vector<Ptr<Vocab>> vocabs_;
  size_t maxLength_;
  bool maxLengthCrop_;
  bool rightLeft_;

  size_t pos_{0};
  
  Ptr<WordAlignment> wordAlignment_;
  
  UPtr<SQLite::Database> db_;
  UPtr<SQLite::Statement> select_;
  
  void fillSQLite();

public:
  CorpusSQLite(Ptr<Config> options, bool translate = false);

  CorpusSQLite(std::vector<std::string> paths,
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

  iterator begin() { return iterator(this); }

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
