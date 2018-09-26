#include "common/config.h"
#include "common/file_stream.h"
#include "common/logging.h"
#include "common/utils.h"
#include "common/version.h"

#include <algorithm>
#include <set>
#include <string>

namespace marian {

size_t Config::seed = (size_t)time(0);

Config::Config(int argc,
               char** argv,
               cli::mode mode /*= cli::mode::training*/,
               bool validate /*= true*/) {
  initialize(argc, argv, mode, validate);
}

Config::Config(const Config& other) : config_(YAML::Clone(other.config_)) {}

void Config::initialize(int argc, char** argv, cli::mode mode, bool validate) {
  auto parser = ConfigParser(argc, argv, mode, validate);
  config_ = parser.getConfig();
  devices_ = parser.getDevices();

  createLoggers(this);

  // set random seed
  if(get<size_t>("seed") == 0)
    seed = (size_t)time(0);
  else
    seed = get<size_t>("seed");

  // load model parameters
  if(mode != cli::mode::translation) {
    auto model = get<std::string>("model");
    if(filesystem::exists(model) && !get<bool>("no-reload")) {
      try {
        if(!get<bool>("ignore-model-config"))
          loadModelParameters(model);
      } catch(std::runtime_error& e) {
        LOG(info, "[config] No model configuration found in model file");
      }
    }
  }
  // if cli::mode::translation
  else {
    auto model = get<std::vector<std::string>>("models")[0];
    try {
      if(!get<bool>("ignore-model-config"))
        loadModelParameters(model);
    } catch(std::runtime_error& ) {
      LOG(info, "[config] No model configuration found in model file");
    }
  }

  log();

  // Log version of Marian that has been used to create the model.
  //
  // Key "version" is present only if loaded from model parameters and is not
  // related to --version flag
  if(has("version")) {
    auto version = get<std::string>("version");

    if(mode == cli::mode::training && version != PROJECT_VERSION_FULL)
      LOG(info,
          "[config] Loaded model has been created with Marian {}, "
          "will be overwritten with current version {} at saving",
          version,
          PROJECT_VERSION_FULL);
    else
      LOG(info,
          "[config] Loaded model has been created with Marian {}",
          version);
  }
  // If this is a newly started training
  else if(mode == cli::mode::training) {
    LOG(info,
        "[config] Model is being created with Marian {}",
        PROJECT_VERSION_FULL);
  }
}

bool Config::has(const std::string& key) const {
  return config_[key];
}

YAML::Node Config::operator[](const std::string& key) const {
  return get(key);
}

YAML::Node Config::get(const std::string& key) const {
  return config_[key];
}

const YAML::Node& Config::get() const {
  return config_;
}

YAML::Node& Config::get() {
  return config_;
}

const std::vector<DeviceId>& Config::getDevices() {
  return devices_;
}

void Config::save(const std::string& name) {
  OutputFileStream out(name);
  (std::ostream&)out << *this;
}

void Config::loadModelParameters(const std::string& name) {
  YAML::Node config;
  io::getYamlFromModel(config, "special:model.yml", name);
  override(config);
}

void Config::loadModelParameters(const void* ptr) {
  YAML::Node config;
  io::getYamlFromModel(config, "special:model.yml", ptr);
  override(config);
}

void Config::override(const YAML::Node& params) {
  for(auto& it : params) {
    config_[it.first.as<std::string>()] = it.second;
  }
}

void Config::log() {
  YAML::Emitter out;
  cli::OutputYaml(config_, out);
  std::string configString = out.c_str();

  // print YAML prepending each line with [config]
  std::vector<std::string> results;
  utils::split(configString, results, "\n");
  for(auto& r : results)
    LOG(info, "[config] {}", r);
}

}  // namespace marian
