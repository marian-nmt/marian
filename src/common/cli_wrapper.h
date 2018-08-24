#pragma once

#include "3rd_party/CLI/CLI.hpp"
#include "3rd_party/yaml-cpp/yaml.h"
#include "common/definitions.h"
#include "common/logging.h"
#include "common/some_type.h"

// TODO: remove
#include "common/cli_helper.h"

#include <iostream>
#include <map>
#include <string>

namespace marian {
namespace cli {

// TODO: remove
template <typename S, typename T>
S &operator<<(S &s, const std::vector<T> &v) {
  for(auto &x : v)
    s << x << " ";
  return s;
}

namespace validators {
const CLI::detail::ExistingFileValidator file_exists;
const CLI::detail::ExistingDirectoryValidator dir_exists;
const CLI::detail::ExistingPathValidator path_exists;

const CLI::detail::NonexistentPathValidator path_not_exists;

typedef CLI::Range range;
}

class CLIWrapper {
private:
  // Stores option variables
  std::map<std::string, Ptr<some_type>> vars_;
  // Stores option objects
  std::map<std::string, CLI::Option *> opts_;
  // Command-line argument parser
  Ptr<CLI::App> app_;
  // Stores options as YAML object
  YAML::Node config_;

  // Name for the current option group
  std::string currentGroup_{""};

  // Print failure message on error
  static std::string failureMessage(const CLI::App *app, const CLI::Error &e);

  std::string keyName(const std::string &args) const {
    return std::get<1>(CLI::detail::get_names(CLI::detail::split_names(args)))
        .front();
  }

public:
  CLIWrapper();

  virtual ~CLIWrapper() {}

  template <typename T>
  CLI::Option *add(const std::string &args, const std::string &help, T val) {
    return add_option<T>(keyName(args), args, help, val, true, true);
  }

  template <typename T>
  CLI::Option *add(const std::string &args, const std::string &help) {
    return add_option<T>(keyName(args), args, help, T(), false, true);
  }

  template <typename T>
  CLI::Option *add_nondefault(const std::string &args,
                              const std::string &help) {
    return add_option<T>(keyName(args), args, help, T(), false, false);
  }

  bool has(const std::string &key) const;

  template <typename T>
  T get(const std::string &key) const {
    ABORT_IF(
        vars_.count(key) == 0, "An option with key '{}' is not defined", key);
    return vars_.at(key)->as<T>();
  }

  void startGroup(const std::string &name) { currentGroup_ = name; }
  void endGroup() { currentGroup_ = ""; }

  Ptr<CLI::App> app() { return app_; }

  YAML::Node getConfig() {
    return config_;
  }

  YAML::Node getConfigWithNewDefaults(const YAML::Node& node) const {
    YAML::Node yaml = YAML::Clone(config_);
    // iterate requested default values
    for(auto it : node) {
      auto key = it.first.as<std::string>();
      // if we have an option and but it was not specified on command-line
      if(vars_.count(key) > 0 && opts_.at(key)->empty()) {
        yaml[key] = YAML::Clone(it.second);
      }
    }
    return yaml;
  }

private:
  template <
      typename T,
      CLI::enable_if_t<!CLI::is_bool<T>::value && !CLI::is_vector<T>::value,
                       CLI::detail::enabler> = CLI::detail::dummy>
  CLI::Option *add_option(const std::string &key,
                          const std::string &args,
                          const std::string &help,
                          T val,
                          bool defaulted,
                          bool addToConfig) {
    //std::cerr << "CLI::add(" << key << ") " << std::endl;

    if(addToConfig)
      config_[key] = val;
    vars_.insert(std::make_pair(key, std::make_shared<some_type>(val)));

    CLI::callback_t fun = [this, key](CLI::results_t res) {
      //std::cerr << "CLI::callback(" << key << ") " << std::endl;
      auto &var = vars_[key]->as<T>();
      auto ret = CLI::detail::lexical_cast(res[0], var);
      config_[key] = var;
      return ret;
    };

    auto opt = app_->add_option(args, fun, help, defaulted);
    opt->type_name(CLI::detail::type_name<T>());
    if(!currentGroup_.empty())
      opt->group(currentGroup_);

    opts_.insert(std::make_pair(key, opt));
    return opts_[key];
  }

  template <typename T,
            CLI::enable_if_t<CLI::is_bool<T>::value,
                             CLI::detail::enabler> = CLI::detail::dummy>
  CLI::Option *add_option(const std::string &key,
                          const std::string &args,
                          const std::string &help,
                          T val,
                          bool defaulted,
                          bool addToConfig) {
    //std::cerr << "CLI::add(" << key << ") as bool" << std::endl;

    if(addToConfig)
      config_[key] = val;
    vars_.insert(std::make_pair(key, std::make_shared<some_type>(val)));

    CLI::callback_t fun = [this, key](CLI::results_t res) {
      //std::cerr << "CLI::callback(" << key << ") " << std::endl;
      config_[key] = !res.empty();
      return true;
    };

    auto opt = app_->add_option(args, fun, help, defaulted);
    opt->type_size(0);
    if(!currentGroup_.empty())
      opt->group(currentGroup_);

    opts_.insert(std::make_pair(key, opt));
    return opts_[key];
  }

  template <typename T,
            CLI::enable_if_t<CLI::is_vector<T>::value,
                             CLI::detail::enabler> = CLI::detail::dummy>
  CLI::Option *add_option(const std::string &key,
                          const std::string &args,
                          const std::string &help,
                          T val,
                          bool defaulted,
                          bool addToConfig) {
    //std::cerr << "CLI::add(" << key << ") as vector" << std::endl;

    if(addToConfig)
      config_[key] = val;
    vars_.insert(std::make_pair(key, std::make_shared<some_type>(val)));

    CLI::callback_t fun = [this, key](CLI::results_t res) {
      //std::cerr << "CLI::callback(" << key << ") " << std::endl;
      auto &vec = vars_[key]->as<T>();
      vec.clear();
      bool ret = true;
      for(const auto &a : res) {
        vec.emplace_back();
        ret &= CLI::detail::lexical_cast(a, vec.back());
      }
      config_[key] = vec;
      return (!vec.empty()) && ret;
    };

    auto opt = app_->add_option(args, fun, help);
    opt->type_name(CLI::detail::type_name<T>())->type_size(-1);
    if(!currentGroup_.empty())
      opt->group(currentGroup_);

    opts_.insert(std::make_pair(key, opt));
    return opts_[key];
  }
};

}  // namespace cli
}  // namespace marian
