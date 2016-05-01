#pragma once


#include "config.h"
#include "types.h"
#include "vocab.h"
#include "loader.h"
#include "scorer.h"
#include "logging.h"
  
class Weights;
  
class God {
  public:
        
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
    
    static std::vector<ScorerPtr> GetScorers(size_t);
    static std::vector<std::string> GetScorerNames();
    static std::map<std::string, float>& GetScorerWeights();
    
    static void CleanUp();
    
    void LoadWeights(const std::string& path);
    
  private:
    God& NonStaticInit(int argc, char** argv);

    static God instance_;
    Config config_;
    
    std::vector<std::unique_ptr<Vocab>> sourceVocabs_;
    std::unique_ptr<Vocab> targetVocab_;
    
    std::map<std::string, LoaderPtr> loaders_;
    std::map<std::string, float> weights_;
    
    std::shared_ptr<spdlog::logger> info_;
    std::shared_ptr<spdlog::logger> progress_;
};

