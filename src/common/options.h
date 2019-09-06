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
 
  // constructor with one or more key-value pairs
  // New<Options>("var1", val1, "var2", val2, ...)
  template <typename T, typename... Args>
  Options(const std::string& key, T value, Args&&... moreArgs) : Options() {
    set(key, value, std::forward<Args>(moreArgs)...);
  }

  // constructor that clones and zero or more updates
  // options->with("var1", val1, "var2", val2, ...)
  template <typename... Args>
  Ptr<Options> with(Args&&... args) const {
    auto options = New<Options>(*this);
    options->set(std::forward<Args>(args)...);
    return options;
  }

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

  // set multiple
  // options->set("var1", val1, "var2", val2, ...)
  template <typename T, typename... Args>
  void set(const std::string& key, T value, Args&&... moreArgs) {
    set(key, value);
    set(std::forward<Args>(moreArgs)...);
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
