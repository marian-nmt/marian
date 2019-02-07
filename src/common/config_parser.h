#pragma once

#include "3rd_party/yaml-cpp/yaml.h"
#include "common/cli_wrapper.h"
#include "common/definitions.h"

// TODO: why are these needed by a config parser? Can they be removed for Linux
// as well?
#ifndef _WIN32
#include <sys/ioctl.h>
#include <unistd.h>
#endif

namespace marian {

namespace cli {
enum struct mode { training, translation, scoring, server };
}  // namespace cli

/**
 * @brief Command-line options parser
 *
 * New options and aliases should be defined within `addOptions*` methods.
 */
class ConfigParser {
public:
  ConfigParser(int argc, char** argv, cli::mode mode, bool validate = false)
      : modeServer_(mode == cli::mode::server),
        mode_(mode == cli::mode::server ? cli::mode::translation : mode) {
    parseOptions(argc, argv, validate);
  }

  /**
   * @brief Parse command-line options
   *
   * Options are parsed in the following order, later config options overwrite
   * earlier:
   *  * predefined default values
   *  * options from the config files provided with --config, from left to right
   *  * options from the model config file, e.g. model.npz.yml
   *  * aliases expanded into options, e.g. --best-deep
   *  * options provided as command-line arguments
   *
   * Parsed options are available from getConfig().
   *
   * @param argc
   * @param argv
   * @param validate Do or do not validate parsed options
   */
  void parseOptions(int argc, char** argv, bool validate);

  YAML::Node getConfig() const;

private:
  bool modeServer_;
  cli::mode mode_;
  YAML::Node config_;

  // Check if the config contains value for option key
  bool has(const std::string& key) const {
    return (bool)config_[key];
  }

  // Return value for given option key cast to given type.
  // Abort if not set.
  template <typename T>
  T get(const std::string& key) const {
    ABORT_IF(!has(key), "CLI object has no key '{}'", key);
    return config_[key].as<T>();
  }

  void addOptionsGeneral(cli::CLIWrapper&);
  void addOptionsServer(cli::CLIWrapper&);
  void addOptionsModel(cli::CLIWrapper&);
  void addOptionsTraining(cli::CLIWrapper&);
  void addOptionsValidation(cli::CLIWrapper&);
  void addOptionsTranslation(cli::CLIWrapper&);
  void addOptionsScoring(cli::CLIWrapper&);

  void addSuboptionsDevices(cli::CLIWrapper&);
  void addSuboptionsBatching(cli::CLIWrapper&);
  void addSuboptionsInputLength(cli::CLIWrapper&);
  void addSuboptionsULR(cli::CLIWrapper&);
  void expandAliases(cli::CLIWrapper&);

  // Extract paths to all config files found in the config object.
  // Look at --config option and model.npz.yml files.
  std::vector<std::string> findConfigPaths();
  // Load options from config files.
  // Handle environment variables and relative paths.
  YAML::Node loadConfigFiles(const std::vector<std::string>&);
};

}  // namespace marian
