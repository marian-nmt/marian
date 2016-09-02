#pragma once

#include <yaml-cpp/yaml.h>
#include <boost/program_options.hpp>

#include "logging.h"

class Config {
  private:
    YAML::Node config_;
    
  public:
    std::string inputPath;

    bool Has(const std::string& key);
    
    YAML::Node Get(const std::string& key) {
      return config_[key];
    }
    
    template <typename T>
    T Get(const std::string& key) {
      return config_[key].as<T>();
    }
    
    YAML::Node& Get();
    
    void AddOptions(size_t argc, char** argv);
    
    template <class OStream>
    friend OStream& operator<<(OStream& out, const Config& config) {
      out << config.config_;
      return out;
    }
    
    void LogOptions();
};
