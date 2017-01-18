#pragma once

#include <vector>
#include <string>
#include <yaml-cpp/yaml.h>

#include "common/scorer.h"
#include "common/loader.h"
#include "common/logging.h"
#include "common/base_best_hyps.h"

namespace CPU {

class Weights;

class EncoderDecoderLoader : public Loader {
  public:
    EncoderDecoderLoader(const std::string name,
                         const YAML::Node& config);

    virtual void Load();

    virtual ScorerPtr NewScorer(God &god, const size_t taskId);
    BestHypsBase &GetBestHyps(God &god);

  private:
    std::vector<std::unique_ptr<Weights>> weights_;
};

} // namespace CPU
