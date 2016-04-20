#pragma once

#include <boost/program_options.hpp>

#include "types.h"
#include "scorer.h"
#include "dl4mt.h"
#include "vocab.h"
#include "kenlm.h"
#include "logging.h"

namespace po = boost::program_options;
  
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
    
    static Vocab& GetSourceVocab();
    static Vocab& GetTargetVocab();
    static std::vector<ScorerPtr> GetScorers(size_t);
    static std::vector<float>& GetScorerWeights();
    
    static void CleanUp();
    
  private:
    God& NonStaticInit(int argc, char** argv);

    static God instance_;
    po::variables_map vm_;
    
    std::unique_ptr<Vocab> sourceVocab_;
    std::unique_ptr<Vocab> targetVocab_;
    
    typedef std::unique_ptr<Weights> Model;
    typedef std::vector<Model> Models;
    typedef std::vector<Models> ModelsPerDevice;

    ModelsPerDevice modelsPerDevice_;
    std::vector<LM> lms_;
    
    std::vector<ScorerPtr> scorers_;
    std::vector<float> weights_;
};

