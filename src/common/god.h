#pragma once
#include <memory>
#include <iostream>

#include "common/config.h"
#include "common/loader.h"
#include "common/logging.h"
#include "common/scorer.h"
#include "common/types.h"
#include "common/processor/processor.h"

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

    static bool Has(const std::string& key) {
      return Summon().config_.Has(key);
    }

    template <typename T>
    static T Get(const std::string& key) {
      return Summon().config_.Get<T>(key);
    }

    static YAML::Node Get(const std::string& key) {
      return Summon().config_.Get(key);
    }

    static Vocab& GetSourceVocab(size_t i = 0);
    static Vocab& GetTargetVocab();

    static std::istream& GetInputStream();

    static Filter& GetFilter();

    static std::vector<ScorerPtr> GetScorers(size_t);
    static std::vector<std::string> GetScorerNames();
    static std::map<std::string, float>& GetScorerWeights();

    static std::vector<std::string> Preprocess(size_t i, const std::vector<std::string>& input);
    static std::vector<std::string> Postprocess(const std::vector<std::string>& input);

    static void CleanUp();

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
};
