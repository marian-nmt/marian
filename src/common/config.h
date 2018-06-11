#pragma once

#include <sys/ioctl.h>
#include <unistd.h>
#include <boost/program_options.hpp>

#include "3rd_party/yaml-cpp/yaml.h"
#include "3rd_party/cnpy/cnpy.h"
#include "common/config_parser.h"
#include "common/file_stream.h"
#include "common/logging.h"
#include "common/utils.h"

namespace marian {

class Config {
public:
  static size_t seed;

  typedef YAML::Node YamlNode;

  Config(const std::string options,
         ConfigMode mode = ConfigMode::training,
         bool validate = false) {
    std::vector<std::string> sargv;
    Split(options, sargv, " ");
    int argc = sargv.size();

    std::vector<char*> argv(argc);
    for(int i = 0; i < argc; ++i)
      argv[i] = const_cast<char*>(sargv[i].c_str());

    initialize(argc, &argv[0], mode, validate);
  }

  Config(int argc,
         char** argv,
         ConfigMode mode = ConfigMode::training,
         bool validate = true) {
    initialize(argc, argv, mode, validate);
  }

  void initialize(int argc,
                  char** argv,
                  ConfigMode mode = ConfigMode::training,
                  bool validate = true) {
    auto parser = ConfigParser(argc, argv, mode, validate);
    config_ = parser.getConfig();
    devices_ = parser.getDevices();

    createLoggers(this);

    if(get<size_t>("seed") == 0)
      seed = (size_t)time(0);
    else
      seed = get<size_t>("seed");

    if(mode != ConfigMode::translating) {
      if(boost::filesystem::exists(get<std::string>("model"))
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

  Config(const Config& other) : config_(YAML::Clone(other.config_)) {}

  bool has(const std::string& key) const;

  YAML::Node get(const std::string& key) const;
  YAML::Node operator[](const std::string& key) const { return get(key); }

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

  template <typename T>
  void set(const std::string& key, const T& value) {
    config_[key] = value;
  }

  const YAML::Node& get() const;
  YAML::Node& get();

  YAML::Node getModelParameters();
  void loadModelParameters(const std::string& name);

  const std::vector<DeviceId>& getDevices() { return devices_; }

  void save(const std::string& name) {
    OutputFileStream out(name);
    (std::ostream&)out << *this;
  }

  friend std::ostream& operator<<(std::ostream& out, const Config& config) {
    YAML::Emitter outYaml;
    OutputYaml(config.get(), outYaml);
    out << outYaml.c_str();
    return out;
  }

  static void AddYamlToNpz(const YAML::Node&,
                           const std::string&,
                           const std::string&);
  static void AddYamlToNpzItems(const YAML::Node&,
                                const std::string&,
                                std::vector<cnpy::NpzItem>&);

private:
  YAML::Node config_;
  std::vector<DeviceId> devices_;

  static void GetYamlFromNpz(YAML::Node&,
                             const std::string&,
                             const std::string&);

  void override(const YAML::Node& params);

  void log();
};
}
