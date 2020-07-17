#include "options.h"

namespace marian {

Options::Options()
#if FASTOPT
  : fastOptions_(options_)
#endif
{}

Options::Options(const Options& other)
#if FASTOPT
  : options_(YAML::Clone(other.options_)),
    fastOptions_(options_) {}
#else
  : options_(YAML::Clone(other.options_)) {}
#endif

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
#if FASTOPT
  setLazyRebuild();
#endif
}

void Options::merge(const YAML::Node& node, bool overwrite) {
  for(auto it : node)
    if(overwrite || !options_[it.first.as<std::string>()])
      options_[it.first.as<std::string>()] = YAML::Clone(it.second);
#if FASTOPT
  setLazyRebuild();
#endif
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
#if FASTOPT
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
#else
  if(!options_[key]) {
    return false;
  } else {
    auto& node = options_[key];
    if(node.IsSequence())
      return node.size() != 0;
    else if(node.IsScalar()) // numerical values count as non-empty
      return !node.as<std::string>().empty();
    else
      ABORT("Wrong node type");
  }
#endif
}

bool Options::hasAndNotEmpty(const std::string& key) const {
  return hasAndNotEmpty(key.c_str());
}

bool Options::has(const char* const key) const {
#if FASTOPT
  lazyRebuild();
  return fastOptions_.has(key);
#else
  return options_[key];
#endif
}

bool Options::has(const std::string& key) const {
  return has(key.c_str());
}

}
