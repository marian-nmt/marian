// -*- mode: c++; indent-tabs-mode: nil; tab-width: 2 -*-
#pragma once

#include <boost/program_options.hpp>

#include <sys/ioctl.h>
#include <unistd.h>
#include "3rd_party/yaml-cpp/yaml.h"
#include "common/file_stream.h"
#include "common/logging.h"
#include "training/config_parser.h"

namespace marian {

class Config {
public:
  static size_t seed;

  Config(int argc,
         char** argv,
         bool validate = true,
         bool translate = false,
         bool rescore = false) {

    auto parser = ConfigParser(argc, argv, validate, translate, rescore);
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

    if(!translate) {
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

  YAML::Node operator[](const std::string& key) const { return get(key); }

  void override(const YAML::Node& params);

  YAML::Node getModelParameters();
  void loadModelParameters(const std::string& name);
  void saveModelParameters(const std::string& name);

  void OutputRec(const YAML::Node node, YAML::Emitter& out) const;

  template <class OStream>
  friend OStream& operator<<(OStream& out, const Config& config) {
    YAML::Emitter outYaml;
    config.OutputRec(config.get(), outYaml);
    out << outYaml.c_str();
    return out;
  }

  void save(const std::string& name) {
    OutputFileStream out(name);
    (std::ostream&)out << *this;
  }

private:
  YAML::Node config_;
  std::vector<std::string> modelFeatures_;

  void GetYamlFromNpz(YAML::Node&, const std::string&, const std::string&);
  void AddYamlToNpz(const YAML::Node&, const std::string&, const std::string&);

  void log();
};
}
