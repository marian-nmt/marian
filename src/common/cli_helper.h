#pragma once

#include <string>

#include "3rd_party/yaml-cpp/yaml.h"
#include "common/logging.h"

namespace marian {
namespace cli {

// helper to replace environment-variable expressions of the form ${VARNAME} in
// a string
static inline std::string interpolateEnvVars(std::string str) {
  // temporary workaround for MS-internal PhillyOnAzure cluster: warm storage
  // presently has the form /hdfs/VC instead of /{gfs,hdfs}/CLUSTER/VC

  // Catch stdin/stdout and do not process
  if(str == "stdin" || str == "stdout") {
    return str;
  }

#if 1
  if(getenv("PHILLY_JOB_ID")) {
    const char* cluster = getenv("PHILLY_CLUSTER");
    const char* vc = getenv("PHILLY_VC");
    // this environment variable exists when running on the cluster
    if(cluster && vc) {
      static const std::string s_gfsPrefix
          = std::string("/gfs/") + cluster + "/" + vc + "/";
      static const std::string s_hdfsPrefix
          = std::string("/hdfs/") + cluster + "/" + vc + "/";
      if(str.find(s_gfsPrefix) == 0)
        str = std::string("/hdfs/") + vc + "/" + str.substr(s_gfsPrefix.size());
      else if(str.find(s_hdfsPrefix) == 0)
        str = std::string("/hdfs/") + vc + "/"
              + str.substr(s_hdfsPrefix.size());
    }
  }
#endif
  for(;;) {
    const auto pos = str.find("${");
    if(pos == std::string::npos)
      return str;
    const auto epos = str.find("}", pos + 2);
    ABORT_IF(epos == std::string::npos,
             "interpolate-env-vars option: ${{ without matching }} in '{}'",
             str.c_str());
    // isolate the variable name
    const auto var = str.substr(pos + 2, epos - (pos + 2));
    const auto val = getenv(var.c_str());
    ABORT_IF(!val,
             "interpolate-env-vars option: environment variable '{}' not "
             "defined in '{}'",
             var.c_str(),
             str.c_str());
    // replace it; then try again for further replacements
    str = str.substr(0, pos) + val + str.substr(epos + 1);
  }
}

// Helper to implement interpolate-env-vars and relative-paths options
static inline void processPaths(
    YAML::Node& node,
    const std::function<std::string(std::string)>& TransformPath,
    const std::set<std::string>& PATHS,
    bool isPath = false,
    const std::string parentKey = "") {
  // For a scalar node (leaves in the config), just transform the path
  if(isPath && node.IsScalar()) {
    std::string nodePath = node.as<std::string>();
    if(!nodePath.empty())
      node = TransformPath(nodePath);
  }
  // For a sequence node, recursively iterate each value
  else if(node.IsSequence()) {
    for(auto&& sub : node) {
      processPaths(sub, TransformPath, PATHS, isPath);

      // Exception for the shortlist option, which keeps a path and three numbers;
      // we want to process the path only and keep the rest untouched
      if(isPath && parentKey == "shortlist")
        break;
    }
  }
  // For a map node that is not a path, recursively iterate each value
  else if(!isPath && node.IsMap()) {
    for(auto&& sub : node) {
      std::string key = sub.first.as<std::string>();
      // Exception for the sqlite option, which has a special value of 'temporary'
      if(key == "sqlite" && sub.second.as<std::string>() == "temporary")
        continue;
      processPaths(sub.second, TransformPath, PATHS, PATHS.count(key) > 0, key);
    }
  }
}

// helper to convert a YAML node recursively into a string
//
// TODO: create a helper function that converts a YAML node into a string
// without an emitter; consider extracting YAML-related helper functions to a
// separate file
// TODO: Look for Frank's function that does that.
static void OutputYaml(const YAML::Node node, YAML::Emitter& out) {
  std::set<std::string> sorter;
  switch(node.Type()) {
    case YAML::NodeType::Null: out << node; break;
    case YAML::NodeType::Scalar: out << node; break;
    case YAML::NodeType::Sequence:
      out << YAML::BeginSeq;
      for(auto&& n : node)
        OutputYaml(n, out);
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
        OutputYaml(node[key], out);
      }
      out << YAML::EndMap;
      break;
    case YAML::NodeType::Undefined: out << node; break;
  }
}

// Change relative paths to absolute paths relative to the config file's
// directory
void makeAbsolutePaths(YAML::Node& config,
                       const std::string& configPath,
                       const std::set<std::string>& PATHS);

}  // namespace cli
}  // namespace marian
