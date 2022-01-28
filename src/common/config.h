#pragma once

#include "3rd_party/yaml-cpp/yaml.h"
#include "common/cli_helper.h"
#include "common/config_parser.h"
#include "common/io.h"
#include "common/options.h"

// TODO: why are these needed by a config parser? Can they be removed for Linux as well?
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
//
// TODO: At the moment Config is an intermediate object used to get an Options object from
// ConfigParser by calling parseOptions(). This should be either renamed and heavily refactorized or
// (better) be build into ConfigParser.
class Config {
public:
  static size_t seed;

  typedef YAML::Node YamlNode;

  Config(ConfigParser const& cp);
  // TODO: remove mode from this class
  Config(int argc,
         char** argv,
         cli::mode mode = cli::mode::training,
         bool validate = true);

  Config(const Config& other);
  Config(const Options& options);

  void initialize(ConfigParser const& cp);

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
  bool loadModelParameters(const std::string& name);
  bool loadModelParameters(const void* ptr);

  std::vector<DeviceId> getDevices(size_t myMPIRank = 0, size_t numRanks = 1);

  void save(const std::string& name);

  friend std::ostream& operator<<(std::ostream& out, const Config& config);

  static std::vector<DeviceId> getDevices(Ptr<Options> options,
                                          size_t myMPIRank = 0,
                                          size_t numMPIProcesses = 1);
private:
  YAML::Node config_;

  // Add options overwritting values for existing ones
  void override(const YAML::Node& params);

  void log();

};

/**
 * Parse the command line options.
 *
 * @param argc number of arguments passed to the program
 * @param argv array of command-line arguments
 * @param mode change the set of available command-line options, e.g. training, translation, etc.
 * @param validate validate parsed options and abort on failure
 *
 * @return parsed options
 */
Ptr<Options> parseOptions(int argc,
                          char** argv,
                          cli::mode mode,
                          bool validate = true);

}  // namespace marian
