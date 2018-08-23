#pragma once

#include "3rd_party/yaml-cpp/yaml.h"
#include "common/cli_wrapper.h"
#include "common/definitions.h"
#include "common/file_stream.h"
#include "common/logging.h"

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

// try to determine the width of the terminal
uint16_t guess_terminal_width(uint16_t max_width = 180);

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
  cli::CLIWrapper cli_;
  YAML::Node config_;

  bool has(const std::string& key) const;

  template <typename T>
  T get(const std::string& key) const {
    return cli_.get<T>(key);
  }

  void addOptionsCommon(cli::CLIWrapper&);
  void addOptionsModel(cli::CLIWrapper&);
  void addOptionsTraining(cli::CLIWrapper&);
  void addOptionsRescore(cli::CLIWrapper&);
  void addOptionsValid(cli::CLIWrapper&);
  void addOptionsTranslate(cli::CLIWrapper&);

  void validateOptions() const;
  void validateDevices() const;

  // change relative paths to absolute paths relative to the config file's
  // directory
  void makeAbsolutePaths(const std::vector<std::string>&);

  std::vector<std::string> loadConfigPaths();
};
}  // namespace marian
