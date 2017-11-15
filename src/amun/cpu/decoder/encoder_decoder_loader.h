#pragma once

#include <vector>
#include <string>
#include <yaml-cpp/yaml.h>

#include "common/scorer.h"
#include "common/loader.h"
#include "common/logging.h"
#include "common/base_best_hyps.h"

namespace amunmt {
namespace CPU {

namespace dl4mt {
class Weights;
}

namespace Nematus {
class Weights;
}

class EncoderDecoderLoader : public Loader {
  public:
    EncoderDecoderLoader(const std::string name,
                         const YAML::Node& config);

    virtual void Load(const God& god);

    virtual ScorerPtr NewScorer(const God &god, const DeviceInfo &deviceInfo) const;
    BestHypsBasePtr GetBestHyps(const God &god, const DeviceInfo &deviceInfo) const;

  private:
    std::vector<std::unique_ptr<dl4mt::Weights>> dl4mtModels_;
    std::vector<std::unique_ptr<Nematus::Weights>> nematusModels_;
};

} // namespace CPU
}
