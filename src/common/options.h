#pragma once

#include <sstream>
#include <string>
#include "common/definitions.h"

#include "3rd_party/yaml-cpp/yaml.h"

#define YAML_REGISTER_TYPE(registered, type)                \
  namespace YAML {                                          \
  template <>                                               \
  struct convert<registered> {                              \
    static Node encode(const registered& rhs) {             \
      type value = static_cast<type>(rhs);                  \
      return Node(value);                                   \
    }                                                       \
    static bool decode(const Node& node, registered& rhs) { \
      type value = node.as<type>();                         \
      rhs = static_cast<registered>(value);                 \
      return true;                                          \
    }                                                       \
  };                                                        \
  }

namespace marian {

/**
 * Container for options stored as key-value pairs. Keys are unique strings.
 */
class Options {
protected:
  YAML::Node options_;

public:
  Options();
  Options(const Options& other);
  // Options(ConfigParser& cp, int argc, char** argv, bool validate);

  /**
   * @brief Return a copy of the object that can be safely modified.
   */
  Options clone() const;

  YAML::Node& getYaml();

  const YAML::Node& getYaml() const;

  void parse(const std::string& yaml);

  /**
   * @brief Splice options from a YAML node
   *
   * By default, only options with keys that do not already exist in options_ are extracted from
   * node. These options are cloned if overwirte is true.
   *
   * @param node a YAML node to transfer the options from
   * @param overwrite overwrite all options
   */
  void merge(const YAML::Node& node, bool overwrite = false);
  void merge(Ptr<Options> options);

  std::string str();

  template <typename T>
  void set(const std::string& key, T value) {
    options_[key] = value;
  }

  template <typename T>
  T get(const std::string& key) const {
    ABORT_IF(!has(key), "Required option '{}' has not been set", key);
    return options_[key].as<T>();
  }

  template <typename T>
  T get(const std::string& key, T defaultValue) const {
    if(has(key))
      return options_[key].as<T>();
    else
      return defaultValue;
  }

  /**
   * @brief Check if a sequence or string option is defined and nonempty
   *
   * Aborts if the option does not store a sequence or string value. Returns false if an option with
   * the given key does not exist.
   *
   * @param key option name
   *
   * @return true if the option is defined and is a nonempty sequence or string
   */
  bool hasAndNotEmpty(const std::string& key) const;

  bool has(const std::string& key) const;
};

}  // namespace marian
