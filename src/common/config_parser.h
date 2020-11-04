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
enum struct mode { training, translation, scoring, server, embedding };
}  // namespace cli

/**
 * @brief Command-line options parser
 *
 * New options and aliases should be defined within `addOptions*` methods.
 * ... unless they are specific to certain executables.
 * In that case, use a pattern like this (e.g., for a server):
 * int main(int argc, char* argv[]) {
 *   ConfigParser cp(cli::mode::translation);
 *   cp.addOption<int>("--port", // option name
 *                     "Server Options", // option group name
 *                     "Port for server.", // help string
 *                     5678); // default value
 *   auto opts = cp.parseOptions(argc,argv,true); // 'true' for validation
 *   ...
 *
 *
 */
class ConfigParser {
public:

  ConfigParser(cli::mode mode);

  ConfigParser(int argc, char** argv, cli::mode mode, bool validate = false)
    : ConfigParser(mode) {
    parseOptions(argc, argv, validate);
  }

  template<typename T>
  ConfigParser&
  addOption(const std::string& args,
            const std::string& group,
            const std::string& help,
            const T val) {
    std::string previous_group = cli_.switchGroup(group);
    cli_.add<T>(args,help,val);
    cli_.switchGroup(previous_group);
    return *this;
  }

  template<typename T>
  ConfigParser&
  addOption(const std::string& args,
            const std::string& group,
            const std::string& help,
            const T val,
            const T implicit_val) {
    std::string previous_group = cli_.switchGroup(group);
    cli_.add<T>(args,help,val)->implicit_val(implicit_val);
    cli_.switchGroup(previous_group);
    return *this;
  }

  template<typename T>
  ConfigParser&
  addOption(const std::string& args,
            const std::string& group,
            const std::string& help) {
    std::string previous_group = cli_.switchGroup(group);
    cli_.add<T>(args,help);
    cli_.switchGroup(previous_group);
    return *this;
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
   * @return (YAML::Node const&)config_
   */

  Ptr<Options> parseOptions(int argc, char** argv, bool validate);
  YAML::Node const& getConfig() const;
  cli::mode getMode() const;
  std::string const& cmdLine() const;
private:
  cli::CLIWrapper cli_;
  cli::mode mode_;
  YAML::Node config_;
  std::string cmdLine_;

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
  void addOptionsEmbedding(cli::CLIWrapper&);

  void addAliases(cli::CLIWrapper&);

  void addSuboptionsDevices(cli::CLIWrapper&);
  void addSuboptionsBatching(cli::CLIWrapper&);
  void addSuboptionsInputLength(cli::CLIWrapper&);
  void addSuboptionsTSV(cli::CLIWrapper&);
  void addSuboptionsULR(cli::CLIWrapper&);
  void addSuboptionsQuantization(cli::CLIWrapper&);

  // Extract paths to all config files found in the config object.
  // Look at --config option and model.npz.yml files.
  std::vector<std::string> findConfigPaths();
  // Load options from config files.
  // Handle environment variables and relative paths.
  YAML::Node loadConfigFiles(const std::vector<std::string>&);
};

}  // namespace marian
