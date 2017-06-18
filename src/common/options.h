#pragma once

#include <string>
#include <sstream>
#include "common/definitions.h"
#include "3rd_party/yaml-cpp/yaml.h"

namespace marian {

class Options {
  protected:
    YAML::Node options_;

  public:
    YAML::Node& getOptions() {
      return options_;
    }

    void parse(const std::string& yaml) {
      auto node = YAML::Load(yaml);
      for(auto it : node)
        options_[it.first.as<std::string>()] = it.second;
    }

    void merge(Ptr<Options> options) {
      merge(options->getOptions());
    }

    void merge(YAML::Node& node) {
      for(auto it : node)
        if(!options_[it.first.as<std::string>()])
          options_[it.first.as<std::string>()] = it.second;
    }

    std::string str() {
      std::stringstream ss;
      ss << options_;
      return ss.str();
    }

    template <typename T>
    void set(const std::string& key, T value) {
      options_[key] = value;
    }

    template <typename T>
    T get(const std::string& key) {
      UTIL_THROW_IF2(!has(key), "Required option \"" << key << "\" has not been set");
      return options_[key].as<T>();
    }

    template <typename T>
    T get(const std::string& key, T defaultValue) {
      if(has(key))
        return options_[key].as<T>();
      else
        return defaultValue;
    }

    bool has(const std::string& key) const {
      return options_[key];
    }
};

}