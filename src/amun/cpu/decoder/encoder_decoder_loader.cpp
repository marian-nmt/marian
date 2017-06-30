#include "cpu/decoder/encoder_decoder_loader.h"

#include <vector>
#include <yaml-cpp/yaml.h>

#include "common/god.h"
#include "cpu/decoder/best_hyps.h"
#include "cpu/dl4mt/encoder_decoder.h"
#include "cpu/nematus/encoder_decoder.h"

using namespace std;

namespace amunmt {
namespace CPU {

EncoderDecoderLoader::EncoderDecoderLoader(
  const std::string name,
  const YAML::Node& config)
  : Loader(name, config)
{}

void EncoderDecoderLoader::Load(const God&) {
  std::string path = Get<std::string>("path");
  std::string type = Get<std::string>("type");

  LOG(info)->info("Loading model {}", path);
  LOG(info)->info("Model type: {}", type);
  if (type == "nematus2") {
    nematusModels_.emplace_back(new Nematus::Weights(path, 0));
  } else {
    dl4mtModels_.emplace_back(new dl4mt::Weights(path, 0));
  }
}

ScorerPtr EncoderDecoderLoader::NewScorer(const God &god, const DeviceInfo&) const {
  size_t tab = Has("tab") ? Get<size_t>("tab") : 0;
  std::string type = Get<std::string>("type");
  if (type == "nematus2") {
    return ScorerPtr(new Nematus::EncoderDecoder(god, name_, config_,
                                              tab, *nematusModels_[0]));
  }
  return ScorerPtr(new dl4mt::EncoderDecoder(god, name_, config_,
                                             tab, *dl4mtModels_[0]));
}

BestHypsBasePtr EncoderDecoderLoader::GetBestHyps(const God &god, const DeviceInfo &deviceInfo) const {
  return BestHypsBasePtr(new CPU::BestHyps(god));
}

}
}
