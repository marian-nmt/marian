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
#include "common/search.h"

#include "scorer.h"
#include "loader_factory.h"

using namespace std;

namespace amunmt {

God::God()
 : threadIncr_(0)
{
}

God::~God()
{
}

God& God::Init(const std::string& options) {
  std::vector<std::string> args = boost::program_options::split_unix(options);
  int argc = args.size() + 1;
  char* argv[argc];
  argv[0] = const_cast<char*>("bogus");
  for (int i = 1; i < argc; ++i) {
    argv[i] = const_cast<char*>(args[i-1].c_str());
  }
  return Init(argc, argv);
}

God& God::Init(int argc, char** argv) {
  info_ = spdlog::stderr_logger_mt("info");
  info_->set_pattern("[%c] (%L) %v");

  progress_ = spdlog::stderr_logger_mt("progress");
  progress_->set_pattern("%v");

  config_.AddOptions(argc, argv);
  config_.LogOptions();

  if (Get("source-vocab").IsSequence()) {
    for (auto sourceVocabPath : Get<std::vector<std::string>>("source-vocab")) {
      sourceVocabs_.emplace_back(new Vocab(sourceVocabPath));
    }
  } else {
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

void God::LoadScorers() {
  LOG(info) << "Loading scorers...";
#ifdef CUDA
  size_t gpuThreads = God::Get<size_t>("gpu-threads");
  auto devices = God::Get<std::vector<size_t>>("devices");
  if (gpuThreads > 0 && devices.size() > 0) {
    for (auto&& pair : config_.Get()["scorers"]) {
      std::string name = pair.first.as<std::string>();
      gpuLoaders_.emplace(name, LoaderFactory::Create(*this, name, pair.second, "GPU"));
    }
  }
#endif
  size_t cpuThreads = God::Get<size_t>("cpu-threads");
  if (cpuThreads) {
    for (auto&& pair : config_.Get()["scorers"]) {
      std::string name = pair.first.as<std::string>();
      cpuLoaders_.emplace(name, LoaderFactory::Create(*this, name, pair.second, "CPU"));
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

  if (Has("bpe") && !Get<bool>("no-debpe")) {
    LOG(info) << "De-BPE output";
    postprocessors_.emplace_back(new BPE());
  }
}

Vocab& God::GetSourceVocab(size_t i) const {
  return *sourceVocabs_[i];
}

Vocab& God::GetTargetVocab() const {
  return *targetVocab_;
}

const Filter& God::GetFilter() const {
  return *filter_;
}

std::istream& God::GetInputStream() const {
  return *inputStream_;
}

OutputCollector& God::GetOutputCollector() const {
  return outputCollector_;
}

std::vector<ScorerPtr> God::GetScorers(const DeviceInfo &deviceInfo) const {
  std::vector<ScorerPtr> scorers;

  if (deviceInfo.deviceType == CPUDevice) {
    for (auto&& loader : cpuLoaders_ | boost::adaptors::map_values)
      scorers.emplace_back(loader->NewScorer(*this, deviceInfo));
  } else {
    for (auto&& loader : gpuLoaders_ | boost::adaptors::map_values)
      scorers.emplace_back(loader->NewScorer(*this, deviceInfo));
  }
  return scorers;
}

BestHypsBasePtr God::GetBestHyps(const DeviceInfo &deviceInfo) const {
  if (deviceInfo.deviceType == CPUDevice) {
    return cpuLoaders_.begin()->second->GetBestHyps(*this);
  } else {
    return gpuLoaders_.begin()->second->GetBestHyps(*this);
  }
}

std::vector<std::string> God::GetScorerNames() const {
  std::vector<std::string> scorerNames;
  for(auto&& name : cpuLoaders_ | boost::adaptors::map_keys)
    scorerNames.push_back(name);
  for(auto&& name : gpuLoaders_ | boost::adaptors::map_keys)
    scorerNames.push_back(name);
  return scorerNames;
}

const std::map<std::string, float>& God::GetScorerWeights() const {
  return weights_;
}

std::vector<std::string> God::Preprocess(size_t i, const std::vector<std::string>& input) const {
  std::vector<std::string> processed = input;
  if (preprocessors_.size() >= i + 1) {
    for (const auto& processor : preprocessors_[i]) {
      processed = processor->Preprocess(processed);
    }
  }
  return processed;
}

std::vector<std::string> God::Postprocess(const std::vector<std::string>& input) const {
  std::vector<std::string> processed = input;
  for (const auto& processor : postprocessors_) {
    processed = processor->Postprocess(processed);
  }
  return processed;
}

void God::CleanUp() {
  for (Loaders::value_type& loader : cpuLoaders_) {
     loader.second.reset(nullptr);
  }
  for (Loaders::value_type& loader : gpuLoaders_) {
     loader.second.reset(nullptr);
  }
}

DeviceInfo God::GetNextDevice() const
{
  DeviceInfo ret;

  size_t cpuThreads = God::Get<size_t>("cpu-threads");
  ret.deviceType = (threadIncr_ < cpuThreads) ? CPUDevice : GPUDevice;

  if (ret.deviceType == CPUDevice) {
    ret.threadInd = threadIncr_;
  }
  else {
    size_t threadIncrGPU = threadIncr_ - cpuThreads;
    size_t gpuThreads = Get<size_t>("gpu-threads");
    std::vector<size_t> devices = Get<std::vector<size_t>>("devices");

    ret.threadInd = threadIncrGPU / devices.size();

    size_t deviceInd = threadIncrGPU % devices.size();
    assert(deviceInd < devices.size());
    ret.deviceId = devices[deviceInd];

    amunmt_UTIL_THROW_IF2(ret.threadInd >= gpuThreads, "Too many GPU threads");
  }

  ++threadIncr_;

  return ret;
}

Search &God::GetSearch() const
{
  Search *obj;

  {
    boost::shared_lock<boost::shared_mutex> read_lock(accessLock_);
    obj = search_.get();
    if (obj) {
      return *obj;
    }
  }

  boost::unique_lock<boost::shared_mutex> lock(accessLock_);
  obj = new Search(*this);
  search_.reset(obj);

  assert(obj);
  return *obj;
}

}

