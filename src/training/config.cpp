#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <set>
#include <string>

#include "3rd_party/cnpy/cnpy.h"
#include "common/file_stream.h"
#include "common/logging.h"
#include "training/config.h"


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

void Config::OutputRec(const YAML::Node node, YAML::Emitter& out) const {
  // std::set<std::string> flow = { "devices" };
  std::set<std::string> sorter;
  switch(node.Type()) {
    case YAML::NodeType::Null: out << node; break;
    case YAML::NodeType::Scalar: out << node; break;
    case YAML::NodeType::Sequence:
      out << YAML::BeginSeq;
      for(auto&& n : node)
        OutputRec(n, out);
      out << YAML::EndSeq;
      break;
    case YAML::NodeType::Map:
      for(auto& n : node)
        sorter.insert(n.first.as<std::string>());
      out << YAML::BeginMap;
      for(auto& key : sorter) {
        out << YAML::Key;
        out << key;
        out << YAML::Value;
        // if(flow.count(key))
        // out << YAML::Flow;
        OutputRec(node[key], out);
      }
      out << YAML::EndMap;
      break;
    case YAML::NodeType::Undefined: out << node; break;
  }
}


void Config::log() {
  YAML::Emitter out;
  OutputRec(config_, out);
  std::string conf = out.c_str();

  std::vector<std::string> results;
  boost::algorithm::split(results, conf, boost::is_any_of("\n"));
  for(auto& r : results)
    LOG(config)->info(r);
}

void Config::override(const YAML::Node& params) {
  for(auto& it : params) {
    config_[it.first.as<std::string>()] = it.second;
  }
}

YAML::Node Config::getModelParameters() {
  YAML::Node modelParams;
  for(auto& key : modelFeatures_)
    modelParams[key] = config_[key];
  return modelParams;
}

void Config::loadModelParameters(const std::string& name) {
  YAML::Node config;
  GetYamlFromNpz(config, "special:model.yml", name);
  override(config);
}

void Config::GetYamlFromNpz(YAML::Node& yaml,
                            const std::string& varName,
                            const std::string& fName) {
  yaml = YAML::Load(cnpy::npz_load(fName, varName).data);
}

void Config::saveModelParameters(const std::string& name) {
  AddYamlToNpz(getModelParameters(), "special:model.yml", name);
}

void Config::AddYamlToNpz(const YAML::Node& yaml,
                          const std::string& varName,
                          const std::string& fName) {
  YAML::Emitter out;
  OutputRec(yaml, out);
  unsigned shape = out.size() + 1;
  cnpy::npz_save(fName, varName, out.c_str(), &shape, 1, "a");
}
}
