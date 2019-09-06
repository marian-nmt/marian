#include "options.h"

namespace marian {
  Options::Options() {}

  Options::Options(const Options& other)
    : options_(YAML::Clone(other.options_)) {}

  Options Options::clone() const {
    return Options(*this);
  }

  YAML::Node& Options::getYaml() {
    return options_;
  }

  const YAML::Node& Options::getYaml() const {
    return options_;
  }

  void Options::parse(const std::string& yaml) {
    auto node = YAML::Load(yaml);
    for(auto it : node)
      options_[it.first.as<std::string>()] = YAML::Clone(it.second);
  }

  void Options::merge(const YAML::Node& node, bool overwrite) {
    for(auto it : node)
      if(overwrite || !options_[it.first.as<std::string>()])
        options_[it.first.as<std::string>()] = YAML::Clone(it.second);
  }

  void Options::merge(Ptr<Options> options) {
    merge(options->getYaml());
  }

  std::string Options::str() {
    std::stringstream ss;
    ss << options_;
    return ss.str();
  }

  bool Options::hasAndNotEmpty(const std::string& key) const {
    if(!has(key)) {
      return false;
    }
    if(options_[key].IsSequence()) {
      return options_[key].size() != 0;
    }
    try {
      return !options_[key].as<std::string>().empty();
    } catch(const YAML::BadConversion& e) {
      ABORT("Option '{}' is neither a sequence nor a text");
    }
    return false;
  }

  bool Options::has(const std::string& key) const {
    return options_[key];
  }
}
