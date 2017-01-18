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

class Weights;
class Vocab;
class Filter;
class InputFileStream;

class God {
  public:
    virtual ~God();

    static God& Init(const std::string&);
    static God& Init(int argc, char** argv);

    static God& Summon() {
      return instance_;
    }

    bool Has(const std::string& key) {
      return config_.Has(key);
    }

    template <typename T>
    static T Get(const std::string& key) {
      return Summon().config_.Get<T>(key);
    }

    YAML::Node Get(const std::string& key) {
      return config_.Get(key);
    }

    Vocab& GetSourceVocab(size_t i = 0);
    Vocab& GetTargetVocab();

    std::istream& GetInputStream();
    OutputCollector& GetOutputCollector();

    Filter& GetFilter();

    BestHypsBase &GetBestHyps(size_t threadId);

    std::vector<ScorerPtr> GetScorers(size_t);
    std::vector<std::string> GetScorerNames();
    std::map<std::string, float>& GetScorerWeights();

    std::vector<std::string> Preprocess(size_t i, const std::vector<std::string>& input);
    std::vector<std::string> Postprocess(const std::vector<std::string>& input);

    void CleanUp();

    void LoadWeights(const std::string& path);

  private:
    God& NonStaticInit(int argc, char** argv);

    void LoadScorers();
    void LoadFiltering();
    void LoadPrePostProcessing();

    static God instance_;
    Config config_;

    std::vector<std::unique_ptr<Vocab>> sourceVocabs_;
    std::unique_ptr<Vocab> targetVocab_;

    std::unique_ptr<Filter> filter_;

    std::vector<std::vector<PreprocessorPtr>> preprocessors_;
    std::vector<PostprocessorPtr> postprocessors_;

    std::map<std::string, LoaderPtr> cpuLoaders_;
    std::map<std::string, LoaderPtr> gpuLoaders_;
    std::map<std::string, float> weights_;

    std::shared_ptr<spdlog::logger> info_;
    std::shared_ptr<spdlog::logger> progress_;

    std::unique_ptr<InputFileStream> inputStream_;
    OutputCollector outputCollector_;

};
