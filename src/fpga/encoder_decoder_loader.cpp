#include "encoder_decoder_loader.h"
#include "encoder_decoder.h"

namespace amunmt {
namespace FPGA {

EncoderDecoderLoader::EncoderDecoderLoader(const std::string name,
                     const YAML::Node& config)
:Loader(name, config)
{

}

void EncoderDecoderLoader::Load(const God &god)
{

}

ScorerPtr EncoderDecoderLoader::NewScorer(const God &god, const DeviceInfo &deviceInfo) const
{
  size_t d = deviceInfo.deviceId;
  size_t tab = Has("tab") ? Get<size_t>("tab") : 0;

  EncoderDecoder *ed = new EncoderDecoder(god, name_, config_, tab, *weights_, context_);
  return ScorerPtr(ed);
}

BestHypsBasePtr EncoderDecoderLoader::GetBestHyps(const God &god) const
{

}


}
}
