#pragma once

#include "3rd_party/yaml-cpp/yaml.h"
#include "common/cli_helper.h"
#include "common/config_parser.h"
#include "common/io.h"

// TODO: why are these needed by a config parser? Can they be removed for Linux
// as well?
#ifndef _WIN32
#include <sys/ioctl.h>
#include <unistd.h>
#endif

namespace marian {

// TODO: Finally refactorize Config, Options, ConfigParser and ConfigValidator
// classes.
//
// TODO: The problem is that there are many config classes in here, plus
// "configuration" can refer to the high-level concept of the entire program's
// configuration, and/or any of its representations. Avoidthe term "config" and
// always qualify it what kind of config, e.g. new Options instance.
//
// TODO: What is not clear is the different config levels as there are classes
// for:
//  - parsing cmd-line options
//  - representing a set of options
//  - interpreting these options in the context of Marian
// It is not clear which class does what, which class knows what.
class Config {
public:
  static size_t seed;

  typedef YAML::Node YamlNode;

  // TODO: remove mode from this class
  Config(int argc,
         char** argv,
         cli::mode mode = cli::mode::training,
         bool validate = true);

  Config(const Config& other);

  void initialize(int argc, char** argv, cli::mode mode, bool validate);

  bool has(const std::string& key) const;

  YAML::Node operator[](const std::string& key) const;
  YAML::Node get(const std::string& key) const;

  template <typename T>
  T get(const std::string& key) const {
    return config_[key].as<T>();
  }

  template <typename T>
  T get(const std::string& key, const T& dflt) const {
    if(has(key))
      return config_[key].as<T>();
    else
      return dflt;
  }

  const YAML::Node& get() const;
  YAML::Node& get();

  template <typename T>
  void set(const std::string& key, const T& value) {
    config_[key] = value;
  }

  YAML::Node getModelParameters();
  void loadModelParameters(const std::string& name);
  void loadModelParameters(const void* ptr);

  // @TODO: remove this accessor or move to a more appropriate class
  const std::vector<DeviceId>& getDevices();

  void save(const std::string& name);

  friend std::ostream& operator<<(std::ostream& out, const Config& config) {
    YAML::Emitter outYaml;
    cli::OutputYaml(config.get(), outYaml);
    out << outYaml.c_str();
    return out;
  }

private:
  YAML::Node config_;
  std::vector<DeviceId> devices_;

  // Add options overwritting values for existing ones
  void override(const YAML::Node& params);

  void log();
};
}  // namespace marian
