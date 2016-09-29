#pragma once
#include <memory>
#include <iostream>

#include "common/config.h"
#include "common/loader.h"
#include "common/logging.h"
#include "common/scorer.h"
#include "common/types.h"

class Weights;
class Vocab;
class Processor;

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

    static std::istream& GetInputStream() {
      return *Summon().inStrm;
    }

    static std::vector<ScorerPtr> GetScorers(size_t);
    static std::vector<std::string> GetScorerNames();
    static std::map<std::string, float>& GetScorerWeights();

    std::vector<std::string> Preprocess(const std::vector<std::string>& input);
    std::vector<std::string> Postprocess(const std::vector<std::string>& input);

    static void CleanUp();

    void LoadWeights(const std::string& path);

  private:
    God& NonStaticInit(int argc, char** argv);

    static God instance_;
    Config config_;

    std::vector<std::unique_ptr<Vocab>> sourceVocabs_;
    std::unique_ptr<Vocab> targetVocab_;

    std::vector<std::unique_ptr<Processor>> processors_;

    std::map<std::string, LoaderPtr> loaders_;
    std::map<std::string, float> weights_;

    std::shared_ptr<spdlog::logger> info_;
    std::shared_ptr<spdlog::logger> progress_;

    std::istream *inStrm;
};

