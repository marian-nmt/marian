#pragma once

#include <yaml-cpp/yaml.h>
#include <boost/program_options.hpp>

class Config {
  public:
    Config(int argc, char** argv, bool validate = true) {
      addOptions(argc, argv, validate);
    }

    bool has(const std::string& key) const;

    YAML::Node get(const std::string& key) const;

    template <typename T>
    T get(const std::string& key) const {
      return config_[key].as<T>();
    }

    const YAML::Node& get() const;
    YAML::Node& get();

    YAML::Node operator[](const std::string& key) const {
      return get(key);
    }

    void addOptions(int argc, char** argv, bool validate);
    void log();
    void validate() const;

    template <class OStream>
    friend OStream& operator<<(OStream& out, const Config& config) {
      out << config.config_;
      return out;
    }

  private:
    std::string inputPath;
    YAML::Node config_;
};
