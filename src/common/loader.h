#pragma once

#include <vector>
#include <yaml-cpp/yaml.h>

#include "scorer.h"
#include "common/base_best_hyps.h"

class Loader {
  public:
    Loader(const std::string& name,
           const YAML::Node& config)
    : name_(name), config_(YAML::Clone(config))
    {}

    virtual ~Loader() {};

    virtual void Load() = 0;

    bool Has(const std::string& key) {
      return config_[key];
    }

    template <typename T>
    T Get(const std::string& key) {
      return config_[key].as<T>();
    }

    virtual ScorerPtr NewScorer(size_t) = 0;
    virtual BestHypsType GetBestHyps() = 0;

    const std::string& GetName() {
      return name_;
    }

  protected:
    const std::string name_;
    const YAML::Node config_;
};

typedef std::unique_ptr<Loader> LoaderPtr;
