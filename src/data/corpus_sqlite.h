#pragma once

#include <fstream>
#include <iostream>
#include <random>

#include "common/definitions.h"
#include "common/file_stream.h"
#include "common/options.h"
#include "data/alignment.h"
#include "data/batch.h"
#include "data/corpus_base.h"
#include "data/dataset.h"
#include "data/vocab.h"

#include <SQLiteCpp/SQLiteCpp.h>
#include <SQLiteCpp/sqlite3/sqlite3.h>

static void SQLiteRandomSeed(sqlite3_context* context,
                             int argc,
                             sqlite3_value** argv) {
  if(argc == 1 && sqlite3_value_type(argv[0]) == SQLITE_INTEGER) {
    const int seed = sqlite3_value_int(argv[0]);
    static std::default_random_engine eng(seed);
    std::uniform_int_distribution<> unif;
    const int result = unif(eng);
    sqlite3_result_int(context, result);
  } else {
    sqlite3_result_error(context, "Invalid", 0);
  }
}

namespace marian {
namespace data {

class CorpusSQLite : public CorpusBase {
private:
  UPtr<SQLite::Database> db_;
  UPtr<SQLite::Statement> select_;

  void fillSQLite();

  size_t seed_;

public:
  // @TODO: check if translate can be replaced by an option in options
  CorpusSQLite(Ptr<Options> options, bool translate = false, size_t seed = Config::seed);

  CorpusSQLite(const std::vector<std::string>& paths,
               const std::vector<Ptr<Vocab>>& vocabs,
               Ptr<Options> options,
               size_t seed = Config::seed);

  Sample next() override;

  void shuffle() override;

  void reset() override;

  void restore(Ptr<TrainingState>) override;

  iterator begin() override { return iterator(this); }

  iterator end() override { return iterator(); }

  std::vector<Ptr<Vocab>>& getVocabs() override { return vocabs_; }

  batch_ptr toBatch(const std::vector<Sample>& batchVector) override {
    size_t batchSize = batchVector.size();

    std::vector<size_t> sentenceIds;

    std::vector<int> maxDims;
    for(auto& ex : batchVector) {
      if(maxDims.size() < ex.size())
        maxDims.resize(ex.size(), 0);
      for(size_t i = 0; i < ex.size(); ++i) {
        if(ex[i].size() > (size_t)maxDims[i])
          maxDims[i] = (int)ex[i].size();
      }
      sentenceIds.push_back(ex.getId());
    }

    std::vector<Ptr<SubBatch>> subBatches;
    for(size_t j = 0; j < maxDims.size(); ++j) {
      subBatches.emplace_back(New<SubBatch>(batchSize, maxDims[j], vocabs_[j]));
    }

    std::vector<size_t> words(maxDims.size(), 0);
    for(size_t i = 0; i < batchSize; ++i) {
      for(size_t j = 0; j < maxDims.size(); ++j) {
        for(size_t k = 0; k < batchVector[i][j].size(); ++k) {
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

    if(options_->has("guided-alignment") && alignFileIdx_)
      addAlignmentsToBatch(batch, batchVector);
    if(options_->hasAndNotEmpty("data-weighting") && weightFileIdx_)
      addWeightsToBatch(batch, batchVector);

    return batch;
  }

private:
  void createRandomFunction() {
    sqlite3_create_function(db_->getHandle(),
                            "random_seed",
                            1,
                            SQLITE_UTF8,
                            NULL,
                            &SQLiteRandomSeed,
                            NULL,
                            NULL);
  }
};
}  // namespace data
}  // namespace marian
