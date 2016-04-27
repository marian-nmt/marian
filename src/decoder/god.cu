#include <vector>
#include <sstream>

#include "god.h"
#include "scorer.h"
#include "threadpool.h"
#include "encoder_decoder.h"
#include "language_model.h"
#include "ape_penalty.h"

God God::instance_;

God& God::Init(const std::string& initString) {
  std::vector<std::string> args = po::split_unix(initString);
  int argc = args.size() + 1;
  char* argv[argc];
  argv[0] = const_cast<char*>("dummy");
  for(int i = 1; i < argc; i++)
    argv[i] = const_cast<char*>(args[i-1].c_str());
  return Init(argc, argv);
}

God& God::Init(int argc, char** argv) {
  return Summon().NonStaticInit(argc, argv);
}

God& God::NonStaticInit(int argc, char** argv) {
  info_ = spdlog::stderr_logger_mt("info");
  info_->set_pattern("[%c] (%L) %v");

  progress_ = spdlog::stderr_logger_mt("progress");
  progress_->set_pattern("%v");

  po::options_description general("General options");

  std::vector<size_t> devices;
  std::vector<std::string> modelPaths;
  std::vector<std::string> lmPaths;
  std::vector<std::string> sourceVocabPaths;
  std::string targetVocabPath;

  general.add_options()
    ("model,m", po::value(&modelPaths)->multitoken()->required(),
     "Path to neural translation model(s)")
    ("source,s", po::value(&sourceVocabPaths)->multitoken()->required(),
     "Path to source vocabulary file.")
    ("target,t", po::value(&targetVocabPath)->required(),
     "Path to target vocabulary file.")
    ("ape", po::value<bool>()->zero_tokens()->default_value(false),
     "Add APE-penalty")
    ("lm,l", po::value(&lmPaths)->multitoken(),
     "Path to KenLM language model(s)")
    ("tab-map", po::value(&tabMap_)->multitoken()->default_value(std::vector<size_t>(1, 0), "0"),
     "tab map")
    ("devices,d", po::value(&devices)->multitoken()->default_value(std::vector<size_t>(1, 0), "0"),
     "CUDA device(s) to use, set to 0 by default, "
     "e.g. set to 0 1 to use gpu0 and gpu1. "
     "Implicitly sets minimal number of threads to number of devices.")
    ("threads-per-device", po::value<size_t>()->default_value(1),
     "Number of threads per device, total thread count equals threads x devices")
    ("help,h", po::value<bool>()->zero_tokens()->default_value(false),
     "Print this help message and exit")
  ;

  po::options_description search("Search options");
  search.add_options()
    ("beam-size,b", po::value<size_t>()->default_value(12),
     "Decoding beam-size")
    ("normalize,n", po::value<bool>()->zero_tokens()->default_value(false),
     "Normalize scores by translation length after decoding")
    ("n-best", po::value<bool>()->zero_tokens()->default_value(false),
     "Output n-best list with n = beam-size")
    ("weights,w", po::value(&weights_)->multitoken()->default_value(std::vector<float>(1, 1.0), "1.0"),
     "Model weights (for neural models and KenLM models)")
    ("show-weights", po::value<bool>()->zero_tokens()->default_value(false),
     "Output used weights to stdout and exit")
    ("load-weights", po::value<std::string>(),
     "Load scorer weights from this file")
  ;

  po::options_description kenlm("KenLM specific options");
  kenlm.add_options()
    ("kenlm-batch-size", po::value<size_t>()->default_value(1000),
     "Batch size for batched queries to KenLM")
    ("kenlm-batch-threads", po::value<size_t>()->default_value(4),
     "Concurrent worker threads for batch processing")
  ;

  po::options_description cmdline_options("Allowed options");
  cmdline_options.add(general);
  cmdline_options.add(search);
  cmdline_options.add(kenlm);

  try {
    po::store(po::command_line_parser(argc,argv)
              .options(cmdline_options).run(), vm_);
    po::notify(vm_);
  }
  catch (std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl << std::endl;

    std::cerr << "Usage: " + std::string(argv[0]) +  " [options]" << std::endl;
    std::cerr << cmdline_options << std::endl;
    exit(1);
  }

  if (Get<bool>("help")) {
    std::cerr << "Usage: " + std::string(argv[0]) +  " [options]" << std::endl;
    std::cerr << cmdline_options << std::endl;
    exit(0);
  }

  PrintConfig();
  
  for(auto& sourceVocabPath : sourceVocabPaths)
    sourceVocabs_.emplace_back(new Vocab(sourceVocabPath));
  targetVocab_.reset(new Vocab(targetVocabPath));

  if(devices.empty()) {
    LOG(info) << "empty";
    devices.push_back(0);
  }

  if(tabMap_.size() < modelPaths.size()) {
    // this should be a warning
    LOG(info) << "More neural models than weights, setting missing tabs to 0";
    tabMap_.resize(modelPaths.size(), 0);
  }
  
  // @TODO: handle this better!
  if(weights_.size() < modelPaths.size()) {
    // this should be a warning
    LOG(info) << "More neural models than weights, setting weights to 1.0";
    weights_.resize(modelPaths.size(), 1.0);
  }
  
  if(Get<bool>("ape") && weights_.size() < modelPaths.size() + 1) {
    LOG(info) << "Adding weight for APE-penalty: " << 1.0;
    weights_.resize(modelPaths.size(), 1.0);
  }

  if(weights_.size() < modelPaths.size() + lmPaths.size()) {
    // this should be a warning
    LOG(info) << "More KenLM models than weights, setting weights to 0.0";
    weights_.resize(weights_.size() + lmPaths.size(), 0.0);
  }
  
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

  for(auto& lmPath : lmPaths) {
    LOG(info) << "Loading lm " << lmPath;
    lms_.emplace_back(lmPath, *targetVocab_);
  }
  
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

void God::PrintConfig() {
  LOG(info) << "Options set: ";
  for(auto& entry: instance_.vm_) {
    std::stringstream ss;
    ss << "\t" << entry.first << " = ";
    try {
      for(auto& v : entry.second.as<std::vector<std::string>>())
        ss << v << " ";
    } catch(...) { }
    try {
      for(auto& v : entry.second.as<std::vector<float>>())
        ss << v << " ";
    } catch(...) { }
    try {
      for(auto& v : entry.second.as<std::vector<size_t>>())
        ss << v << " ";
    } catch(...) { }
    try {
      ss << entry.second.as<std::string>();
    } catch(...) { }
    try {
      ss << entry.second.as<bool>() ? "true" : "false";
    } catch(...) { }
    try {
      ss << entry.second.as<size_t>();
    } catch(...) { }
    
    LOG(info) << ss.str();
  }
}