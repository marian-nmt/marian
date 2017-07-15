#pragma once

#include <boost/program_options.hpp>
#include <sys/ioctl.h>
#include <unistd.h>

#include "3rd_party/yaml-cpp/yaml.h"
#include "common/config_parser.h"
#include "common/file_stream.h"
#include "common/logging.h"

namespace marian {

class Config {
public:
  static size_t seed;

  Config(int argc,
         char** argv,
         ConfigMode mode = ConfigMode::training,
         bool validate = true) {

    auto parser = ConfigParser(argc, argv, mode, validate);
    config_ = parser.getConfig();
    createLoggers(this);

    modelFeatures_ = {
        "type",
        "dim-vocabs",
        "dim-emb",
        "dim-rnn",
        "enc-cell",
        "enc-type",
        "enc-cell-depth",
        "enc-depth",
        "dec-depth",
        "dec-cell",
        "dec-cell-base-depth",
        "dec-cell-high-depth",
        //"dec-high-context",
        "skip",
        "layer-normalization",
        "special-vocab",
        "tied-embeddings"
        /*"lexical-table", "vocabs"*/
    };

    if(get<size_t>("seed") == 0)
      seed = (size_t)time(0);
    else
      seed = get<size_t>("seed");

    if(mode != ConfigMode::translating) {
      if(boost::filesystem::exists(get<std::string>("model"))
         && !get<bool>("no-reload")) {
        try {
          loadModelParameters(get<std::string>("model"));
        } catch(std::runtime_error& e) {
          LOG(info)->info("No model settings found in model file");
        }
      }
    } else {
      auto model = get<std::vector<std::string>>("models")[0];
      try {
        loadModelParameters(model);
      } catch(std::runtime_error& e) {
        LOG(info)->info("No model settings found in model file");
      }
    }
    log();
  }

  Config(const Config& other)
      : config_(YAML::Clone(other.config_)),
        modelFeatures_(other.modelFeatures_) {}

  bool has(const std::string& key) const;

  YAML::Node get(const std::string& key) const;
  YAML::Node operator[](const std::string& key) const { return get(key); }

  template <typename T>
  T get(const std::string& key) const {
    return config_[key].as<T>();
  }

  template <typename T>
  void set(const std::string& key, const T& value) {
    config_[key] = value;
  }

  const YAML::Node& get() const;
  YAML::Node& get();

  YAML::Node getModelParameters();
  void loadModelParameters(const std::string& name);
  void saveModelParameters(const std::string& name);

  void save(const std::string& name) {
    OutputFileStream out(name);
    (std::ostream&)out << *this;
  }

  template <class OStream>
  friend OStream& operator<<(OStream& out, const Config& config) {
    YAML::Emitter outYaml;
    OutputYaml(config.get(), outYaml);
    out << outYaml.c_str();
    return out;
  }

private:
  YAML::Node config_;
  std::vector<std::string> modelFeatures_;

  void GetYamlFromNpz(YAML::Node&, const std::string&, const std::string&);
  void AddYamlToNpz(const YAML::Node&, const std::string&, const std::string&);

  void override(const YAML::Node& params);

  void log();
};
}
