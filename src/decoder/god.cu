#include <vector>
#include <sstream>

#include <yaml-cpp/yaml.h>

#include "god.h"
#include "config.h"
#include "scorer.h"
#include "threadpool.h"
#include "encoder_decoder.h"
#include "language_model.h"
#include "ape_penalty.h"

God God::instance_;

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
  
  for(auto sourceVocabPath : Get<std::vector<std::string>>("source"))
    sourceVocabs_.emplace_back(new Vocab(sourceVocabPath));
  targetVocab_.reset(new Vocab(Get<std::string>("target")));

  auto modelPaths = Get<std::vector<std::string>>("model");
  
  tabMap_ =  Get<std::vector<size_t>>("tab-map");
  if(tabMap_.size() < modelPaths.size()) {
    // this should be a warning
    LOG(info) << "More neural models than tabs, setting missing tabs to 0";
    tabMap_.resize(modelPaths.size(), 0);
  }
  
  // @TODO: handle this better!
  weights_ = Get<std::vector<float>>("weights");
  if(weights_.size() < modelPaths.size()) {
    // this should be a warning
    LOG(info) << "More neural models than weights, setting weights to 1.0";
    weights_.resize(modelPaths.size(), 1.0);
  }
  
  if(Get<bool>("ape") && weights_.size() < modelPaths.size() + 1) {
    LOG(info) << "Adding weight for APE-penalty: " << 1.0;
    weights_.resize(modelPaths.size(), 1.0);
  }

  //if(weights_.size() < modelPaths.size() + lmPaths.size()) {
  //  // this should be a warning
  //  LOG(info) << "More KenLM models than weights, setting weights to 0.0";
  //  weights_.resize(weights_.size() + lmPaths.size(), 0.0);
  //}
  
  if(Has("load-weights")) {
    LoadWeights(Get<std::string>("load-weights"));
  }
  
  if(Get<bool>("show-weights")) {
    LOG(info) << "Outputting weights and exiting";
    for(size_t i = 0; i < weights_.size(); ++i) {
      std::cout << "F" << i << "= " << weights_[i] << std::endl;
    }
    exit(0);
  }
  
  auto devices = Get<std::vector<size_t>>("devices");
  modelsPerDevice_.resize(devices.size());
  {
    ThreadPool devicePool(devices.size());
    for(auto& modelPath : modelPaths) {
      for(size_t i = 0; i < devices.size(); ++i) {
        devicePool.enqueue([i, &devices, &modelPath, this]{
          LOG(info) << "Loading model " << modelPath << " onto gpu" << devices[i];
          cudaSetDevice(devices[i]);
          modelsPerDevice_[i].emplace_back(new Weights(modelPath, devices[i]));
        });
      }
    }
  }

  //for(auto& lmPath : lmPaths) {
  //  LOG(info) << "Loading lm " << lmPath;
  //  lms_.emplace_back(lmPath, *targetVocab_);
  //}
  
  return *this;
}

Vocab& God::GetSourceVocab(size_t i) {
  return *(Summon().sourceVocabs_[i]);
}

Vocab& God::GetTargetVocab() {
  return *Summon().targetVocab_;
}

std::vector<ScorerPtr> God::GetScorers(size_t threadId) {
  size_t deviceId = threadId % Summon().modelsPerDevice_.size();
  size_t device = Summon().modelsPerDevice_[deviceId][0]->GetDevice();
  cudaSetDevice(device);
  std::vector<ScorerPtr> scorers;
  size_t i = 0;
  for(auto& m : Summon().modelsPerDevice_[deviceId])
    scorers.emplace_back(new EncoderDecoder(*m, Summon().tabMap_[i++]));
  if(God::Get<bool>("ape"))
    scorers.emplace_back(new ApePenalty(Summon().tabMap_[i++]));
  for(auto& lm : Summon().lms_)
    scorers.emplace_back(new LanguageModel(lm));
  return scorers;
}

std::vector<float>& God::GetScorerWeights() {
  return Summon().weights_;
}

std::vector<size_t>& God::GetTabMap() {
  return Summon().tabMap_;
}

// clean up cuda vectors before cuda context goes out of scope
void God::CleanUp() {
  for(auto& models : Summon().modelsPerDevice_)
    for(auto& m : models)
      m.reset(nullptr);
}

void God::LoadWeights(const std::string& path) {
  LOG(info) << "Reading weights from " << path;
  std::ifstream fweights(path.c_str());
  std::string name;
  float weight;
  size_t i = 0;
  weights_.clear();
  while(fweights >> name >> weight) {
    LOG(info) << " > F" << i << "= " << weight; 
    weights_.push_back(weight);
    i++;
  }
}
