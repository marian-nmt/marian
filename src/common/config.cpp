#include "3rd_party/cnpy/cnpy.h"
#include "common/config.h"
#include "common/file_stream.h"
#include "common/logging.h"

#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <set>
#include <string>

namespace marian {

size_t Config::seed = (size_t)time(0);

bool Config::has(const std::string& key) const {
  return config_[key];
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

void Config::log() {
  YAML::Emitter out;
  OutputYaml(config_, out);
  std::string configString = out.c_str();

  std::vector<std::string> results;
  boost::algorithm::split(results, configString, boost::is_any_of("\n"));
  for(auto& r : results)
    LOG(info, "[config] {}", r);
}

void Config::override(const YAML::Node& params) {
  for(auto& it : params) {
    config_[it.first.as<std::string>()] = it.second;
  }
}

void Config::loadModelParameters(const std::string& name) {
  YAML::Node config;
  GetYamlFromNpz(config, "special:model.yml", name);
  override(config);
}

void Config::GetYamlFromNpz(YAML::Node& yaml,
                            const std::string& varName,
                            const std::string& fName) {
  yaml = YAML::Load(cnpy::npz_load(fName, varName)->data());
}

// helper to serialize a YAML::Node to a Yaml string in a 0-terminated character vector
static std::vector<char> asYamlCharVector(const YAML::Node node)
{
  YAML::Emitter out;
  OutputYaml(node, out);
  return std::vector<char>(out.c_str(), out.c_str() + strlen(out.c_str()) + 1);
}

void Config::AddYamlToNpz(const YAML::Node& yaml,
                          const std::string& varName,
                          const std::string& fName) {
  // YAML::Node's Yaml representation is saved as a 0-terminated char vector to the NPZ file
  auto yamlCharVector = asYamlCharVector(yaml);
  unsigned int shape = yamlCharVector.size();
  cnpy::npz_save(fName, varName, yamlCharVector.data(), &shape, 1, "a");
}

// same as AddYamlToNpz() but adds to an in-memory NpzItem vector instead
void Config::AddYamlToNpzItems(const YAML::Node& yaml,
                               const std::string& varName,
                               std::vector<cnpy::NpzItem>& allItems) {
  auto yamlCharVector = asYamlCharVector(yaml);
  allItems.emplace_back(varName, yamlCharVector, std::vector<unsigned int>{ (unsigned int)yamlCharVector.size() });
}
}
