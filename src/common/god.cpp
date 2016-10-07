#include <vector>
#include <sstream>
#include <boost/range/adaptor/map.hpp>

#include <yaml-cpp/yaml.h>

#include "common/god.h"
#include "common/vocab.h"
#include "common/config.h"
#include "common/threadpool.h"
#include "common/file_stream.h"
#include "common/filter.h"
#include "common/processor/bpe.h"
#include "common/utils.h"

#include "scorer.h"
#include "loader_factory.h"

God God::instance_;

God::~God(){}

God& God::Init(const std::string& options) {
  std::vector<std::string> args = boost::program_options::split_unix(options);
  int argc = args.size() + 1;
  char* argv[argc];
  argv[0] = const_cast<char*>("bogus");
  for(int i = 1; i < argc; i++)
    argv[i] = const_cast<char*>(args[i-1].c_str());
  return Init(argc, argv);
}

God& God::Init(int argc, char** argv) {
  return Summon().NonStaticInit(argc, argv);
}

void God::LoadScorers() {
  LOG(info) << "Loading scorers...";
#ifdef CUDA
  size_t gpuThreads = God::Get<size_t>("gpu-threads");
  auto devices = God::Get<std::vector<size_t>>("devices");
  if (gpuThreads > 0 && devices.size() > 0) {
    for (auto&& pair : config_.Get()["scorers"]) {
      std::string name = pair.first.as<std::string>();
      gpuLoaders_.emplace(name, LoaderFactory::Create(name, pair.second, "GPU"));
    }
  }
#endif
  size_t cpuThreads = God::Get<size_t>("cpu-threads");
  if (cpuThreads) {
    for (auto&& pair : config_.Get()["scorers"]) {
      std::string name = pair.first.as<std::string>();
      cpuLoaders_.emplace(name, LoaderFactory::Create(name, pair.second, "CPU"));
    }
  }
}

void God::LoadFiltering() {
  if (!Get<std::vector<std::string>>("softmax-filter").empty()) {
    auto filterOptions = Get<std::vector<std::string>>("softmax-filter");
    std::string alignmentFile = filterOptions[0];
    LOG(info) << "Reading target softmax filter file from " << alignmentFile;
    Filter* filter = nullptr;
    if (filterOptions.size() >= 3) {
      const size_t numNFirst = stoi(filterOptions[1]);
      const size_t maxNumTranslation = stoi(filterOptions[2]);
      filter = new Filter(GetSourceVocab(0),
                          GetTargetVocab(),
                          alignmentFile,
                          numNFirst,
                          maxNumTranslation);
    } else if (filterOptions.size() == 2) {
      const size_t numNFirst = stoi(filterOptions[1]);
      filter = new Filter(GetSourceVocab(0),
                          GetTargetVocab(),
                          alignmentFile,
                          numNFirst);
    } else {
      filter = new Filter(GetSourceVocab(0),
                          GetTargetVocab(),
                          alignmentFile);
    }
    filter_.reset(filter);
  }
}

void God::LoadPrePostProcessing() {
  if (Has("bpe")) {
    if(Get("bpe").IsSequence()) {
      size_t i = 0;
      for(auto bpePath : Get<std::vector<std::string>>("bpe")) {
        LOG(info) << "using bpe: " << bpePath;
        preprocessors_.push_back(std::vector<PreprocessorPtr>());
        preprocessors_[i++].emplace_back(new BPE(bpePath));
      }
    }
    else {
      LOG(info) << "using bpe: " << Get<std::string>("bpe");
        preprocessors_.push_back(std::vector<PreprocessorPtr>());
      if (Get<std::string>("bpe") != "") {
        preprocessors_[0].emplace_back(new BPE(Get<std::string>("bpe")));
      }
    }
  }

  if (Get<bool>("debpe")) {
    LOG(info) << "De-BPE output";
    postprocessors_.emplace_back(new BPE());
  }
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

  LoadScorers();
  LoadFiltering();

  if (Has("input-file")) {
    LOG(info) << "Reading from " << Get<std::string>("input-file");
    inputStream_.reset(new InputFileStream(Get<std::string>("input-file")));
  }
  else {
    LOG(info) << "Reading from stdin";
    inputStream_.reset(new InputFileStream(std::cin));
  }

  LoadPrePostProcessing();

  return *this;
}

Vocab& God::GetSourceVocab(size_t i) {
  return *(Summon().sourceVocabs_[i]);
}

Vocab& God::GetTargetVocab() {
  return *Summon().targetVocab_;
}

Filter& God::GetFilter() {
  return *(Summon().filter_);
}

std::istream& God::GetInputStream() {
  return *Summon().inputStream_;
}

std::vector<ScorerPtr> God::GetScorers(size_t threadId) {
  std::vector<ScorerPtr> scorers;

  size_t cpuThreads = God::Get<size_t>("cpu-threads");

  if (threadId < cpuThreads) {
    for (auto&& loader : Summon().cpuLoaders_ | boost::adaptors::map_values)
      scorers.emplace_back(loader->NewScorer(threadId));
  } else {
    for (auto&& loader : Summon().gpuLoaders_ | boost::adaptors::map_values)
      scorers.emplace_back(loader->NewScorer(threadId - cpuThreads));
  }
  return scorers;
}

std::vector<std::string> God::GetScorerNames() {
  std::vector<std::string> scorerNames;
  for(auto&& name : Summon().cpuLoaders_ | boost::adaptors::map_keys)
    scorerNames.push_back(name);
  return scorerNames;
}

std::map<std::string, float>& God::GetScorerWeights() {
  return Summon().weights_;
}

std::vector<std::string> God::Preprocess(size_t i, const std::vector<std::string>& input) {
  std::vector<std::string> processed = input;
  if (Summon().preprocessors_.size() >= i + 1) {
    for (const auto& processor : Summon().preprocessors_[i]) {
      processed = processor->Preprocess(processed);
    }
  }
  return processed;
}

std::vector<std::string> God::Postprocess(const std::vector<std::string>& input) {
  std::vector<std::string> processed = input;
  for (const auto& processor : Summon().postprocessors_) {
    processed = processor->Postprocess(processed);
  }
  return processed;
}
// clean up cuda vectors before cuda context goes out of scope
void God::CleanUp() {
  for (auto& loader : Summon().cpuLoaders_ | boost::adaptors::map_values) {
     loader.reset(nullptr);
  }
  for (auto& loader : Summon().gpuLoaders_ | boost::adaptors::map_values) {
     loader.reset(nullptr);
  }
}
