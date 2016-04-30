#pragma once

#include <vector>
#include <yaml-cpp/yaml.h>

#include "scorer.h"

class Loader {
  public:
    Loader(const YAML::Node& config)
    : config_(YAML::Clone(config)) {}

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

  protected:
    const YAML::Node config_;
};

typedef std::unique_ptr<Loader> LoaderPtr;
