#include <vector>
#include "encoder_decoder_loader.h"
#include "encoder_decoder.h"
#include "best_hyps.h"
#include "common/threadpool.h"
#include "common/god.h"

namespace amunmt {
namespace GPU {

////////////////////////////////////////////
EncoderDecoderLoader::EncoderDecoderLoader(const std::string name,
                     const YAML::Node& config)
 : Loader(name, config)
{
}

void EncoderDecoderLoader::Load(const God &god) {
  std::string path = Get<std::string>("path");
  std::vector<size_t> devices = god.Get<std::vector<size_t>>("devices");

  size_t maxDeviceId = 0;
  for (size_t i = 0; i < devices.size(); ++i) {
    if (devices[i] > maxDeviceId) {
      maxDeviceId = devices[i];
    }
  }

  ThreadPool devicePool(devices.size());
  weights_.resize(maxDeviceId + 1);

  for(auto d : devices) {
    devicePool.enqueue([d, &path, this] {
        LOG(info->info("Loading model {} onto gpu {}", path, d));
        HANDLE_ERROR(cudaSetDevice(d));
        weights_[d].reset(new Weights(path, config_, d));
      });
  }
}

EncoderDecoderLoader::~EncoderDecoderLoader()
{
  for (size_t d = 0; d < weights_.size(); ++d) {
    const Weights *weights = weights_[d].get();
    if (weights) {
      HANDLE_ERROR(cudaSetDevice(d));
      weights_[d].reset(nullptr);
    }
  }
}

ScorerPtr EncoderDecoderLoader::NewScorer(const God &god, const DeviceInfo &deviceInfo) const {
  size_t d = deviceInfo.deviceId;

  HANDLE_ERROR(cudaSetDevice(d));
  size_t tab = Has("tab") ? Get<size_t>("tab") : 0;
  return ScorerPtr(new EncoderDecoder(god, name_, config_,
                                      tab, *weights_[d]));
}

BestHypsBasePtr EncoderDecoderLoader::GetBestHyps(const God &god, const DeviceInfo &deviceInfo) const {
  BestHypsBasePtr obj(new GPU::BestHyps(god));

  //std::thread::id this_id = std::this_thread::get_id();
  //std::cerr << "deviceInfo=" << deviceInfo << " thread " << this_id << " sleeping...\n";

  return obj;
}

}
}
