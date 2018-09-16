#include "common/config.h"
#include "common/file_stream.h"
#include "common/logging.h"
#include "common/utils.h"

#include <algorithm>
#include <boost/algorithm/string.hpp>
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

  if(get<size_t>("seed") == 0)
    seed = (size_t)time(0);
  else
    seed = get<size_t>("seed");

  if(mode != cli::mode::translation) {
    if(filesystem::exists(get<std::string>("model"))
       && !get<bool>("no-reload")) {
      try {
        if(!get<bool>("ignore-model-config"))
          loadModelParameters(get<std::string>("model"));
      } catch(std::runtime_error& e) {
        LOG(info, "[config] No model configuration found in model file");
      }
    }
  } else {
    auto model = get<std::vector<std::string>>("models")[0];
    try {
      if(!get<bool>("ignore-model-config"))
        loadModelParameters(model);
    } catch(std::runtime_error& e) {
      LOG(info, "[config] No model configuration found in model file");
    }
  }
  log();

  if(has("version"))
    LOG(info,
        "[config] Model created with Marian {}",
        get("version").as<std::string>());
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
  boost::algorithm::split(results, configString, boost::is_any_of("\n"));
  for(auto& r : results)
    LOG(info, "[config] {}", r);
}

}  // namespace marian
