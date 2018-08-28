#pragma once

#include "3rd_party/yaml-cpp/yaml.h"
#include "common/config_parser.h"

namespace marian {

class ConfigValidator
{
private:
  ConfigMode mode_;
  const YAML::Node& config_;

  bool has(const std::string& key) const;

  template <typename T>
  T get(const std::string& key) const {
    return config_[key].as<T>();
  }

  void validateOptions() const;
  void validateDevices() const;

public:
  ConfigValidator(ConfigMode mode, const YAML::Node& config);
  virtual ~ConfigValidator();

  void validate() const;
};

}  // namespace marian
