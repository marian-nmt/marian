#pragma once

#include "common/loader.h"
#include "gpu/dl4mt/model.h"

namespace amunmt {
namespace GPU {

////////////////////////////////////////////
class EncoderDecoderLoader : public Loader {
  public:
    EncoderDecoderLoader(const EncoderDecoderLoader&) = delete;
    EncoderDecoderLoader(const std::string name,
                         const YAML::Node& config);
    virtual ~EncoderDecoderLoader();

    virtual void Load(const God &god);

    virtual ScorerPtr NewScorer(const God &god, const DeviceInfo &deviceInfo) const;
    virtual BestHypsBasePtr GetBestHyps(const God &god, const DeviceInfo &deviceInfo) const;

  private:
    std::vector<std::unique_ptr<Weights>> weights_; // MUST be indexed by gpu id. eg. weights_[2] is for gpu2
};

}
}

