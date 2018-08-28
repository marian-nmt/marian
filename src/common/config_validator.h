#pragma once

#include "3rd_party/yaml-cpp/yaml.h"
#include "common/config_parser.h"

namespace marian {

class ConfigValidator
{
private:
  const YAML::Node& config_;

  bool has(const std::string& key) const;

  template <typename T>
  T get(const std::string& key) const {
    return config_[key].as<T>();
  }

  void validateOptionsTranslation() const;
  void validateOptionsParallelData() const;
  void validateOptionsScoring() const;
  void validateOptionsTraining() const;

public:
  ConfigValidator(const YAML::Node& config);
  virtual ~ConfigValidator();

  void validateOptions(ConfigMode mode) const;
  void validateDevices(ConfigMode mode) const;
};

}  // namespace marian
