#include "encoder_decoder_loader.h"

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

}

BestHypsBasePtr EncoderDecoderLoader::GetBestHyps(const God &god) const
{

}


}
}
