#pragma once

#include <fstream>
#include <iostream>
#include <random>

#include "3rd_party/threadpool.h"
#include "common/definitions.h"
#include "common/file_stream.h"
#include "common/options.h"
#include "data/alignment.h"
#include "data/batch.h"
#include "data/corpus_base.h"
#include "data/dataset.h"
#include "data/vocab.h"

namespace marian {
namespace data {

class Corpus : public CorpusBase {
private:
  std::vector<UPtr<io::TemporaryFile>> tempFiles_;
  std::vector<size_t> ids_;
  
  UPtr<ThreadPool> threadPool_; // thread pool for parallelized data reading

  // for shuffle-in-ram
  bool shuffleInRAM_{false};
  std::vector<std::vector<std::string>> corpusInRAM_; // // [stream][id] full copy of all data files

  void shuffleData(const std::vector<std::string>& paths);

  // for pre-processing
  size_t allCapsEvery_{0};   // if set, convert every N-th input sentence (after randomization) to all-caps (source and target)
  size_t titleCaseEvery_{0}; // ditto for title case (source only)
  void preprocessLine(std::string& line, size_t streamId, bool& altered); // altered => whether the segmentation was altered in marian

public:
  // @TODO: check if translate can be replaced by an option in options
  Corpus(Ptr<Options> options, 
         bool translate = false, 
         size_t seed = Config::seed);

  Corpus(std::vector<std::string> paths,
         std::vector<Ptr<Vocab>> vocabs,
         Ptr<Options> options,
         size_t seed = Config::seed);

  /**
   * @brief Iterates sentence tuples in the corpus.
   *
   * A sentence tuple is skipped with no warning if any sentence in the tuple
   * (e.g. a source or target) is longer than the maximum allowed sentence
   * length in words unless the option "max-length-crop" is provided.
   *
   * @return A tuple representing parallel sentences.
   */
  Sample next() override;

  void shuffle() override;

  void reset() override;

  void restore(Ptr<TrainingState>) override;

  iterator begin() override { return iterator(this); }

  iterator end() override { return iterator(); }

  std::vector<Ptr<Vocab>>& getVocabs() override { return vocabs_; }

  batch_ptr toBatch(const std::vector<Sample>& batchVector) override;
};
}  // namespace data
}  // namespace marian
