#pragma once
#include <memory>
#include <iostream>

#include "common/processor/processor.h"
#include "common/config.h"
#include "common/loader.h"
#include "common/logging.h"
#include "common/scorer.h"
#include "common/types.h"
#include "common/base_best_hyps.h"
#include "common/output_collector.h"
#include "common/vocab.h"
#include "common/threadpool.h"
#include "common/file_stream.h"
#include "common/filter.h"
#include "common/processor/bpe.h"
#include "common/utils.h"

class Weights;
class Vocab;
class Filter;
class InputFileStream;

class God {
  public:
    virtual ~God();

    God& Init(const std::string&);
    God& Init(int argc, char** argv);


    bool Has(const std::string& key) const {
      return config_.Has(key);
    }

    template <typename T>
    T Get(const std::string& key) const {
      return config_.Get<T>(key);
    }

    YAML::Node Get(const std::string& key) const {
      return config_.Get(key);
    }

    Vocab& GetSourceVocab(size_t i = 0);
    Vocab& GetTargetVocab();

    std::istream& GetInputStream();
    OutputCollector& GetOutputCollector() const;

    const Filter& GetFilter() const;

    BestHypsBase &GetBestHyps(size_t threadId) const;

    std::vector<ScorerPtr> GetScorers(size_t) const;
    std::vector<std::string> GetScorerNames() const;
    const std::map<std::string, float>& GetScorerWeights() const;

    std::vector<std::string> Preprocess(size_t i, const std::vector<std::string>& input);
    std::vector<std::string> Postprocess(const std::vector<std::string>& input);

    void CleanUp();

    void LoadWeights(const std::string& path);

  private:
    void LoadScorers();
    void LoadFiltering();
    void LoadPrePostProcessing();

    Config config_;

    std::vector<std::unique_ptr<Vocab>> sourceVocabs_;
    std::unique_ptr<Vocab> targetVocab_;

    std::unique_ptr<const Filter> filter_;

    std::vector<std::vector<PreprocessorPtr>> preprocessors_;
    std::vector<PostprocessorPtr> postprocessors_;

    std::map<std::string, LoaderPtr> cpuLoaders_;
    std::map<std::string, LoaderPtr> gpuLoaders_;
    std::map<std::string, float> weights_;

    std::shared_ptr<spdlog::logger> info_;
    std::shared_ptr<spdlog::logger> progress_;

    std::unique_ptr<InputFileStream> inputStream_;
    mutable OutputCollector outputCollector_;

};
