#pragma once
#include "common/loader.h"

namespace amunmt {
namespace FPGA {

class EncoderDecoderLoader : public Loader
{
  public:
  EncoderDecoderLoader(const std::string name,
                       const YAML::Node& config);

  virtual void Load(const God &god);

  virtual ScorerPtr NewScorer(const God &god, const DeviceInfo &deviceInfo) const;

  virtual BestHypsBasePtr GetBestHyps(const God &god) const;

};


}
}

