#include <vector>
#include <sstream>
#include <boost/range/adaptor/map.hpp>
#include <boost/timer/timer.hpp>

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
#include "common/sentences.h"
#include "common/translation_task.h"
#include "common/logging.h"

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
  Cleanup();
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

  config_.AddOptions(argc, argv);
  info_ = spdlog::stderr_logger_mt("info");
  info_->set_pattern("[%c] (%L) %v");
  set_loglevel(*info_, config_.Get<string>("log-info"));

  progress_ = spdlog::stderr_logger_mt("progress");
  progress_->set_pattern("%v");
  set_loglevel(*progress_, config_.Get<string>("log-progress"));
  
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
    LOG(info)->info("Outputting weights and exiting");
    for(auto && pair : weights_) {
      std::cout << pair.first << "= " << pair.second << std::endl;
    }
    exit(0);
  }

  LoadScorers();
  LoadFiltering();

  returnNBestList_ = Get<bool>("return-alignment")
                   || Get<bool>("return-soft-alignment")
                   || Get<bool>("return-nematus-alignment");

  useFusedSoftmax_ = true;
  if (returnNBestList_ ||
      gpuLoaders_.size() != 1 || // more than 1 scorer
      God::Get<size_t>("beam-size") > 11 // beam size affect shared mem alloc in gLogSoftMax()
      ) {
    useFusedSoftmax_ = false;
  }
  //cerr << "useFusedSoftmax_=" << useFusedSoftmax_ << endl;

  if (Has("input-file")) {
    LOG(info)->info("Reading from {}", Get<std::string>("input-file"));
    inputStream_.reset(new InputFileStream(Get<std::string>("input-file")));
  }
  else {
    LOG(info)->info("Reading from stdin");
    inputStream_.reset(new InputFileStream(std::cin));
  }

  LoadPrePostProcessing();

  size_t totalThreads = GetTotalThreads();
  LOG(info)->info("Total number of threads: {}", totalThreads);
  amunmt_UTIL_THROW_IF2(totalThreads == 0, "Total number of threads is 0");

  pool_.reset(new ThreadPool(totalThreads, totalThreads));

  return *this;
}

void God::Cleanup()
{
  pool_.reset();
  cpuLoaders_.clear();
  gpuLoaders_.clear();
  fpgaLoaders_.clear();
}

void God::LoadScorers() {
  LOG(info)->info("Loading scorers...");
#ifdef CUDA
  size_t gpuThreads = God::Get<size_t>("gpu-threads");
  auto devices = God::Get<std::vector<size_t>>("devices");
  if (gpuThreads > 0 && devices.size() > 0) {
    for (auto&& pair : config_.Get()["scorers"]) {
      std::string name = pair.first.as<std::string>();
      gpuLoaders_.emplace(name, LoaderFactory::Create(*this, name, pair.second, GPUDevice));
    }
  }
#endif
#ifdef HAS_CPU
  size_t cpuThreads = God::Get<size_t>("cpu-threads");
  if (cpuThreads) {
    for (auto&& pair : config_.Get()["scorers"]) {
      std::string name = pair.first.as<std::string>();
      cpuLoaders_.emplace(name, LoaderFactory::Create(*this, name, pair.second, CPUDevice));
    }
  }
#endif
#ifdef HAS_FPGA
  size_t fpgaThreads = God::Get<size_t>("fpga-threads");
  if (fpgaThreads) {
    for (auto&& pair : config_.Get()["scorers"]) {
      std::string name = pair.first.as<std::string>();
      fpgaLoaders_.emplace(name, LoaderFactory::Create(*this, name, pair.second, FPGADevice));
    }
  }
#endif

}

void God::LoadFiltering() {
  if (!Get<std::vector<std::string>>("softmax-filter").empty()) {
    auto filterOptions = Get<std::vector<std::string>>("softmax-filter");
    std::string alignmentFile = filterOptions[0];
    LOG(info)->info("Reading target softmax filter file from {}", alignmentFile);
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
        LOG(info)->info("using bpe: {}", bpePath);
        preprocessors_.push_back(std::vector<PreprocessorPtr>());
        preprocessors_[i++].emplace_back(new BPE(bpePath));
      }
    }
    else {
      LOG(info)->info("using bpe: {}", Get<std::string>("bpe"));
        preprocessors_.push_back(std::vector<PreprocessorPtr>());
      if (Get<std::string>("bpe") != "") {
        preprocessors_[0].emplace_back(new BPE(Get<std::string>("bpe")));
      }
    }
  }

  if (Has("bpe") && !Get<bool>("no-debpe")) {
    LOG(info)->info("De-BPE output");
    postprocessors_.emplace_back(new BPE());
  }
}

Vocab& God::GetSourceVocab(size_t i) const {
  return *sourceVocabs_[i];
}

Vocab& God::GetTargetVocab() const {
  return *targetVocab_;
}

std::shared_ptr<const Filter> God::GetFilter() const {
  return filter_;
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
  }
  else if (deviceInfo.deviceType == GPUDevice) {
    for (auto&& loader : gpuLoaders_ | boost::adaptors::map_values)
      scorers.emplace_back(loader->NewScorer(*this, deviceInfo));
  }
  else if (deviceInfo.deviceType == FPGADevice) {
    for (auto&& loader : fpgaLoaders_ | boost::adaptors::map_values)
      scorers.emplace_back(loader->NewScorer(*this, deviceInfo));
  }
  else {
	amunmt_UTIL_THROW2("Unknown device type:" << deviceInfo);
  }

  return scorers;
}

BestHypsBasePtr God::GetBestHyps(const DeviceInfo &deviceInfo) const {
  if (deviceInfo.deviceType == CPUDevice) {
    return cpuLoaders_.begin()->second->GetBestHyps(*this, deviceInfo);
  }
  else if (deviceInfo.deviceType == GPUDevice) {
    return gpuLoaders_.begin()->second->GetBestHyps(*this, deviceInfo);
  }
  else if (deviceInfo.deviceType == FPGADevice) {
    return fpgaLoaders_.begin()->second->GetBestHyps(*this, deviceInfo);
  }
  else {
	amunmt_UTIL_THROW2("Unknown device type:" << deviceInfo);
  }
}

std::vector<std::string> God::GetScorerNames() const {
  std::vector<std::string> scorerNames;
  for(auto&& name : cpuLoaders_ | boost::adaptors::map_keys)
    scorerNames.push_back(name);
  for(auto&& name : gpuLoaders_ | boost::adaptors::map_keys)
    scorerNames.push_back(name);
  for(auto&& name : fpgaLoaders_ | boost::adaptors::map_keys)
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

DeviceInfo God::GetNextDevice() const
{
  DeviceInfo ret;

  size_t cpuThreads = 0, gpuThreads = 0, fpgaThreads = 0;
  std::vector<size_t> gpuDevices, fpgaDevices;

#ifdef CUDA
  gpuThreads = Get<size_t>("gpu-threads");
  gpuDevices = Get<std::vector<size_t>>("devices");
#endif

#ifdef HAS_CPU
  cpuThreads = God::Get<size_t>("cpu-threads");
#endif

#ifdef HAS_FPGA
  fpgaThreads = Get<size_t>("fpga-threads");
  fpgaDevices = Get<std::vector<size_t>>("fpga-devices");
#endif

  size_t totGPUThreads = gpuThreads * gpuDevices.size();
  size_t totFPGAThreads = fpgaThreads * fpgaDevices.size();

  // start locking
  boost::unique_lock<boost::shared_mutex> lock(accessLock_);

  if (threadIncr_ < cpuThreads) {
    ret.deviceType = CPUDevice;
    ret.threadInd = threadIncr_;
  }
  else if (threadIncr_ < cpuThreads + totGPUThreads) {
    ret.deviceType = GPUDevice;
    size_t threadIncr = threadIncr_ - cpuThreads;

    ret.threadInd = threadIncr / gpuDevices.size();

    size_t deviceInd = threadIncr % gpuDevices.size();
    assert(deviceInd < gpuDevices.size());
    ret.deviceId = gpuDevices[deviceInd];
  }
  else if (threadIncr_ < cpuThreads + totGPUThreads + totFPGAThreads) {
    ret.deviceType = FPGADevice;
    size_t threadIncr = threadIncr_ - cpuThreads - totGPUThreads;

    ret.threadInd = threadIncr / fpgaDevices.size();

    size_t deviceInd = threadIncr % fpgaDevices.size();
    assert(deviceInd < fpgaDevices.size());
    ret.deviceId = fpgaDevices[deviceInd];
  }
  else {
    amunmt_UTIL_THROW2("Too many threads");
  }

  ++threadIncr_;

  return ret;
}

Search &God::GetSearch() const
{
  thread_local Search obj(*this);
  return obj;
}

size_t God::GetTotalThreads() const
{
  size_t totalThreads = 0;

#ifdef CUDA
  size_t gpuThreads = Get<size_t>("gpu-threads");
  auto devices = Get<std::vector<size_t>>("devices");
  LOG(info)->info("Setting GPU thread count to {}", gpuThreads);
  totalThreads += gpuThreads * devices.size();
#endif

#ifdef HAS_CPU
  size_t cpuThreads = Get<size_t>("cpu-threads");
  LOG(info->info("Setting CPU thread count to {}", cpuThreads));
  totalThreads += cpuThreads;
#endif

#ifdef HAS_FPGA
  size_t fpgaThreads = Get<size_t>("fpga-threads");
  auto fpgaDevices = Get<std::vector<size_t>>("fpga-devices");
  LOG(info->info("Setting FPGA thread count to {}", fpgaThreads));
  totalThreads += fpgaThreads * fpgaDevices.size();
#endif

  return totalThreads;
}

}

