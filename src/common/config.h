#pragma once

#include <yaml-cpp/yaml.h>
#include <boost/program_options.hpp>

#include "logging.h"

namespace amunmt {

class Config {
  private:
    YAML::Node config_;
    
  public:
    std::string inputPath;

    bool Has(const std::string& key) const;
    
    YAML::Node Get(const std::string& key) const;
    
    template <typename T>
    T Get(const std::string& key) const {
      return config_[key].as<T>();
    }
    
    const YAML::Node& Get() const;
    
    void AddOptions(size_t argc, char** argv);
    
    template <class OStream>
    friend OStream& operator<<(OStream& out, const Config& config) {
      out << config.config_;
      return out;
    }
    
    void LogOptions();
};

}

