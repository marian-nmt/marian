#pragma once

// @TODO: to be removed when sure it works
#define FASTOPT 1 // for diagnostics, 0 reverts to old behavior

#include <sstream>
#include <string>
#include "common/definitions.h"
#include "3rd_party/yaml-cpp/yaml.h"

#ifdef FASTOPT
#include "common/fastopt.h"
#endif

#define YAML_REGISTER_TYPE(registered, type)                \
namespace YAML {                                            \
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
 * This is not thread-safe and locking is the responsibility of the caller.
 */
class Options {
protected:
  YAML::Node options_;  // YAML options use for parsing, modification and printing
  
#if FASTOPT
  // Only to be modified in lazyRebuild and setLazyRebuild
  mutable FastOpt fastOptions_; // FastOpt used for fast lookup, lazily rebuilt from YYAML whenever required
  mutable bool lazyRebuildPending_{false}; // flag if need to lazily rebuild

  // set flag that a rebuild is required
  void setLazyRebuild() const {
    lazyRebuildPending_ = true;
  }

  // check if rebuild is required, rebuild, unset flag.
  void lazyRebuild() const {
    if(lazyRebuildPending_) {
      FastOpt temp(options_);
      fastOptions_.swap(temp);
      lazyRebuildPending_ = false;
    }
  }
#endif

public:
  Options();
  Options(const Options& other);
 
  // constructor with one or more key-value pairs
  // New<Options>("var1", val1, "var2", val2, ...)
  template <typename T, typename... Args>
  Options(const std::string& key, T value, Args&&... moreArgs) : Options() {
    set(key, value, std::forward<Args>(moreArgs)...);
  }

  Options(const YAML::Node& node) : Options() {
     merge(node);
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

  // Do not allow access to internal YAML object as changes on the outside are difficult to track
  // and mess with the rebuilding of the fast options lookup. Hence only return a clone which guarentees
  // full encapsulation.
  YAML::Node cloneToYamlNode() const;

  void parse(const std::string& yaml);

  /**
   * @brief Splice options from a YAML node
   *
   * By default, only options with keys that do not already exist in options_ are extracted from
   * node. These options are cloned if overwrite is true.
   *
   * @param node a YAML node to transfer the options from
   * @param overwrite overwrite all options
   */
  void merge(const YAML::Node& node, bool overwrite = false);
  void merge(Ptr<Options> options);

  std::string asYamlString();

  template <typename T>
  void set(const std::string& key, T value) {
    options_[key] = value;
#if FASTOPT
    setLazyRebuild();
#endif
  }

  // set multiple
  // options->set("var1", val1, "var2", val2, ...)
  template <typename T, typename... Args>
  void set(const std::string& key, T value, Args&&... moreArgs) {
    set(key, value);
    set(std::forward<Args>(moreArgs)...);
#if FASTOPT
    setLazyRebuild();
#endif
  }

  template <typename T>
  T get(const char* const key) const {
#if FASTOPT
    lazyRebuild();
    ABORT_IF(!has(key), "Required option '{}' has not been set", key);
    return fastOptions_[key].as<T>();
#else
    ABORT_IF(!has(key), "Required option '{}' has not been set", key);
    return options_[key].as<T>();
#endif
  }

  template <typename T>
  T get(const std::string& key) const {
    return get<T>(key.c_str());
  }

  template <typename T>
  T get(const char* const key, T defaultValue) const {
#if FASTOPT
    lazyRebuild();
    if(has(key))
      return fastOptions_[key].as<T>();
#else
    if(has(key))
      return options_[key].as<T>();
#endif
    else
      return defaultValue;
  }

  template <typename T>
  T get(const std::string& key, T defaultValue) const {
    return get<T>(key.c_str(), defaultValue);
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
  bool hasAndNotEmpty(const char* const key) const;
  bool hasAndNotEmpty(const std::string& key) const;

  bool has(const char* const key) const;
  bool has(const std::string& key) const;
};

}  // namespace marian
