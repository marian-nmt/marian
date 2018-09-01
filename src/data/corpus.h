#pragma once

#include <fstream>
#include <iostream>
#include <random>

#include <boost/algorithm/string.hpp>
#include <boost/iterator/iterator_facade.hpp>

#include "common/config.h"
#include "common/definitions.h"
#include "common/file_stream.h"
#include "data/alignment.h"
#include "data/batch.h"
#include "data/corpus_base.h"
#include "data/dataset.h"
#include "data/vocab.h"

namespace marian {
namespace data {

class Corpus : public CorpusBase {
private:
  std::vector<UPtr<TemporaryFile>> tempFiles_;
  std::vector<size_t> ids_;

  void shuffleFiles(const std::vector<std::string>& paths);

public:
  Corpus(Ptr<Config> options, bool translate = false);

  Corpus(std::vector<std::string> paths,
         std::vector<Ptr<Vocab>> vocabs,
         Ptr<Config> options);

  /**
   * @brief Iterates sentence tuples in the corpus.
   *
   * A sentence tuple is skipped with no warning if any sentence in the tuple
   * (e.g. a source or target) is longer than the maximum allowed sentence
   * length in words unless the option "max-length-crop" is provided.
   *
   * @return A tuple representing parallel sentences.
   */
  sample next() override;

  void shuffle() override;

  void reset() override;

  void restore(Ptr<TrainingState>) override;

  iterator begin() override { return iterator(this); }

  iterator end() override { return iterator(); }

  std::vector<Ptr<Vocab>>& getVocabs() override { return vocabs_; }

  batch_ptr toBatch(const std::vector<sample>& batchVector) override {
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
    if(options_->has("data-weighting") && weightFileIdx_)
      addWeightsToBatch(batch, batchVector);

    return batch;
  }
};
}  // namespace data
}  // namespace marian
