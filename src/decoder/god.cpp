#include <vector>
#include <sstream>
#include <boost/range/adaptor/map.hpp>

#include <yaml-cpp/yaml.h>

#include "god.h"
#include "config.h"
#include "scorer.h"
#include "threadpool.h"
#include "file_stream.h"
#include "loader_factory.h"

God God::instance_;

God::~God()
{
  if (inStrm != &std::cin) {
    delete inStrm;
  }
}

God& God::Init(int argc, char** argv) {
  return Summon().NonStaticInit(argc, argv);
}

God& God::NonStaticInit(int argc, char** argv) {
  info_ = spdlog::stderr_logger_mt("info");
  info_->set_pattern("[%c] (%L) %v");

  progress_ = spdlog::stderr_logger_mt("progress");
  progress_->set_pattern("%v");

  config_.AddOptions(argc, argv);
  config_.LogOptions();
  
  if(Get("source-vocab").IsSequence()) {
    for(auto sourceVocabPath : Get<std::vector<std::string>>("source-vocab"))
      sourceVocabs_.emplace_back(new Vocab(sourceVocabPath));
  }
  else {
    sourceVocabs_.emplace_back(new Vocab(Get<std::string>("source-vocab")));    
  }
  targetVocab_.reset(new Vocab(Get<std::string>("target-vocab")));

  weights_ = Get<std::map<std::string, float>>("weights");
    
  if(Get<bool>("show-weights")) {
    LOG(info) << "Outputting weights and exiting";
    for(auto && pair : weights_) {
      std::cout << pair.first << "= " << pair.second << std::endl;
    }
    exit(0);
  }
  
  for(auto&& pair : config_.Get()["scorers"]) {
    std::string name = pair.first.as<std::string>();
    loaders_.emplace(name, LoaderFactory::Create(name, pair.second));
  }
  
  if (config_.inputPath.empty()) {
    std::cerr << "Using cin" << std::endl;
    inStrm = &std::cin;
  }
  else {
    std::cerr << "Using " << config_.inputPath << std::endl;
    inStrm = new std::ifstream(config_.inputPath.c_str());
  }

  return *this;
}

Vocab& God::GetSourceVocab(size_t i) {
  return *(Summon().sourceVocabs_[i]);
}

Vocab& God::GetTargetVocab() {
  return *Summon().targetVocab_;
}

std::vector<ScorerPtr> God::GetScorers(size_t taskId) {
  std::vector<ScorerPtr> scorers;
  for(auto&& loader : Summon().loaders_ | boost::adaptors::map_values)
    scorers.emplace_back(loader->NewScorer(taskId));
  return scorers;
}

std::vector<std::string> God::GetScorerNames() {
  std::vector<std::string> scorerNames;
  for(auto&& name : Summon().loaders_ | boost::adaptors::map_keys)
    scorerNames.push_back(name);
  return scorerNames;
}

std::map<std::string, float>& God::GetScorerWeights() {
  return Summon().weights_;
}

// clean up cuda vectors before cuda context goes out of scope
void God::CleanUp() {
  for(auto& loader : Summon().loaders_ | boost::adaptors::map_values)
     loader.reset(nullptr);
}
