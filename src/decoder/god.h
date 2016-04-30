#pragma once


#include "config.h"
#include "types.h"
#include "vocab.h"
#include "loader.h"
#include "scorer.h"
#include "logging.h"

// this should not be here
#include "kenlm.h"
  
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
    
    static Vocab& GetSourceVocab(size_t i = 0);
    static Vocab& GetTargetVocab();
    static std::vector<ScorerPtr> GetScorers(size_t);
    static std::vector<float>& GetScorerWeights();
    static std::vector<size_t>& GetTabMap();
    
    static void CleanUp();
    
    void LoadWeights(const std::string& path);
    
  private:
    God& NonStaticInit(int argc, char** argv);

    static God instance_;
    Config config_;
    
    std::vector<std::unique_ptr<Vocab>> sourceVocabs_;
    std::unique_ptr<Vocab> targetVocab_;
    
    std::vector<LoaderPtr> loaders_;
    
    std::vector<float> weights_;
    
    std::shared_ptr<spdlog::logger> info_;
    std::shared_ptr<spdlog::logger> progress_;
};

