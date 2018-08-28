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

enum struct ConfigMode {
  training,
  translating,
  rescoring,
};

class ConfigParser {
public:
  ConfigParser(int argc, char** argv, ConfigMode mode, bool validate = false)
      : mode_(mode) {
    parseOptions(argc, argv, validate);
  }

  void parseOptions(int argc, char** argv, bool validate);

  YAML::Node getConfig() const;
  std::vector<DeviceId> getDevices();

private:
  ConfigMode mode_;
  YAML::Node config_;

  void addOptionsGeneral(cli::CLIWrapper&);
  void addOptionsModel(cli::CLIWrapper&);
  void addOptionsTraining(cli::CLIWrapper&);
  void addOptionsValidation(cli::CLIWrapper&);
  void addOptionsTranslation(cli::CLIWrapper&);
  void addOptionsScoring(cli::CLIWrapper&);

  void addSuboptionsDevices(cli::CLIWrapper&);
  void addSuboptionsBatching(cli::CLIWrapper&);
  void addSuboptionsLength(cli::CLIWrapper&);

  // change relative paths to absolute paths relative to the config file's
  // directory
  void makeAbsolutePaths(const std::vector<std::string>&);

  std::vector<std::string> loadConfigPaths();
  YAML::Node loadConfigFiles(const std::vector<std::string>&);
};
}  // namespace marian
