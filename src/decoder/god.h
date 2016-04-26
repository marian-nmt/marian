#pragma once

#include <boost/program_options.hpp>

#include "types.h"
#include "vocab.h"
#include "scorer.h"
#include "logging.h"

// this should not be here
#include "kenlm.h"

namespace po = boost::program_options;
  
class Weights;
  
class God {
  public:
        
    static God& Init(const std::string&);
    static God& Init(int argc, char** argv);

    static God& Summon() {
      return instance_;
    }

    static bool Has(const std::string& key) {
      return instance_.vm_.count(key) > 0;
    }

    template <typename T>
    static T Get(const std::string& key) {
      return instance_.vm_[key].as<T>();
    }
    
    static Vocab& GetSourceVocab(size_t i = 0);
    static Vocab& GetTargetVocab();
    static std::vector<ScorerPtr> GetScorers(size_t);
    static std::vector<float>& GetScorerWeights();
    
    static void CleanUp();
    static void PrintConfig();
    
    void LoadWeights(const std::string& path);
    
  private:
    God& NonStaticInit(int argc, char** argv);

    static God instance_;
    po::variables_map vm_;
    
    std::vector<std::unique_ptr<Vocab>> sourceVocabs_;
    std::unique_ptr<Vocab> targetVocab_;
    
    typedef std::unique_ptr<Weights> Model;
    typedef std::vector<Model> Models;
    typedef std::vector<Models> ModelsPerDevice;

    ModelsPerDevice modelsPerDevice_;
    std::vector<LM> lms_;
    
    std::vector<ScorerPtr> scorers_;
    std::vector<float> weights_;
    
    std::shared_ptr<spdlog::logger> info_;
    std::shared_ptr<spdlog::logger> progress_;
};

