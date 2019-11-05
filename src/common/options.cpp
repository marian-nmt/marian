#include "options.h"

namespace marian {
  Options::Options() 
  : fastOptions_(options_) {}

  Options::Options(const Options& other)
    : options_(YAML::Clone(other.options_)),
      fastOptions_(options_) {}

  Options Options::clone() const {
    return Options(*this); // fastOptions_ get set in constructor above
  }

  YAML::Node Options::cloneToYamlNode() const {
    return YAML::Clone(options_); // Do not give access to internal YAML object
  }

  void Options::parse(const std::string& yaml) {
    auto node = YAML::Load(yaml);
    for(auto it : node)
      options_[it.first.as<std::string>()] = YAML::Clone(it.second);
    setLazyRebuild();
  }

  void Options::merge(const YAML::Node& node, bool overwrite) {
    for(auto it : node)
      if(overwrite || !options_[it.first.as<std::string>()])
        options_[it.first.as<std::string>()] = YAML::Clone(it.second);
    setLazyRebuild();
  }

  void Options::merge(Ptr<Options> options) {
    merge(options->options_);
  }

  std::string Options::asYamlString() {
    std::stringstream ss;
    ss << options_;
    return ss.str();
  }

  bool Options::hasAndNotEmpty(const char* const key) const {
    lazyRebuild();
    if(!fastOptions_.has(key)) {
      return false;
    } else {
      auto& node = fastOptions_[key];
      if(node.isSequence())
        return node.size() != 0;
      else if(node.isScalar()) // numerical values count as non-empty
        return !node.as<std::string>().empty();
      else
        ABORT("Wrong node type");
    }
  }

  bool Options::hasAndNotEmpty(const std::string& key) const {
    return hasAndNotEmpty(key.c_str());
  }

  bool Options::has(const char* const key) const {
    lazyRebuild();
    return fastOptions_.has(key);
  }

  bool Options::has(const std::string& key) const {
    return has(key.c_str());
  }
}
