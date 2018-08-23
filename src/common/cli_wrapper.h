#pragma once

#include "3rd_party/CLI/CLI.hpp"
#include "3rd_party/yaml-cpp/yaml.h"
#include "common/some_type.h"
#include "common/logging.h"

// TODO: remove
#include "common/cli_helper.h"

#include <map>
#include <string>
#include <iostream>

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
  std::map<std::string, std::shared_ptr<some_type>> vars_;
  // Stores option objects
  std::map<std::string, std::shared_ptr<CLI::Option>> opts_;
  // Command-line argument parser
  std::shared_ptr<CLI::App> app_;
  // Stores options as YAML object
  YAML::Node config_;

  // Name for the current option group
  std::string currentGroup_{""};

  // Print failure message on error
  static std::string failureMessage(const CLI::App *app, const CLI::Error &e);

public:
  CLIWrapper();

  virtual ~CLIWrapper() {}

  template <
      typename T,
      CLI::enable_if_t<!CLI::is_bool<T>::value && !CLI::is_vector<T>::value,
                       CLI::detail::enabler> = CLI::detail::dummy>
  std::shared_ptr<CLI::Option> add(const std::string &key,
                                   const std::string &args,
                                   const std::string &help,
                                   T val = T()) {
    std::cerr << "CLI::add(" << key << ") " << std::endl;

    config_[key] = val;
    vars_.insert(std::make_pair(key, std::make_shared<some_type>(val)));

    CLI::callback_t fun = [this, key](CLI::results_t res) {
      auto& var = vars_[key]->as<T>();
      auto ret = CLI::detail::lexical_cast(res[0], var);
      config_[key] = var;
      return ret;
    };

    std::shared_ptr<CLI::Option> opt(app_->add_option(args, fun, help, true));
    opt->type_name(CLI::detail::type_name<T>());
    if(!currentGroup_.empty())
      opt->group(currentGroup_);
    opts_.insert(std::make_pair(key, opt));
    return opts_[key];
  }

  template <typename T,
            CLI::enable_if_t<CLI::is_bool<T>::value,
                             CLI::detail::enabler> = CLI::detail::dummy>
  std::shared_ptr<CLI::Option> add(const std::string &key,
                                   const std::string &args,
                                   const std::string &help,
                                   T val = T()) {
    std::cerr << "CLI::add(" << key << ") as bool" << std::endl;

    config_[key] = false;
    vars_.insert(std::make_pair(key, std::make_shared<some_type>(val)));

    CLI::callback_t fun = [this, key](CLI::results_t res) {
      config_[key] = !res.empty();
      return true;
    };

    std::shared_ptr<CLI::Option> opt(app_->add_option(args, fun, help, true));
    opt->type_size(0);
    if(!currentGroup_.empty())
      opt->group(currentGroup_);
    opts_.insert(std::make_pair(key, opt));
    return opts_[key];
  }

  template <typename T,
            CLI::enable_if_t<CLI::is_vector<T>::value,
                             CLI::detail::enabler> = CLI::detail::dummy>
  std::shared_ptr<CLI::Option> add(const std::string &key,
                                   const std::string &args,
                                   const std::string &help,
                                   T val = T()) {
    std::cerr << "CLI::add(" << key << ") as vector" << std::endl;

    vars_.insert(std::make_pair(key, std::make_shared<some_type>(val)));
    config_[key] = val;

    CLI::callback_t fun = [this, key](CLI::results_t res) {
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

    std::shared_ptr<CLI::Option> opt(app_->add_option(args, fun, help));
    opt->type_name(CLI::detail::type_name<T>())->type_size(-1);
    if(!currentGroup_.empty())
      opt->group(currentGroup_);
    opts_.insert(std::make_pair(key, opt));
    return opts_[key];
  }

  template <typename T>
  std::shared_ptr<CLI::Option> getOption(const std::string &key) {
    return opts_[key];
  }

  bool has(const std::string &key) const;

  template <typename T>
  T get(const std::string &key) const {
    ABORT_IF(vars_.count(key) == 0, "An options with key '{}' does not exist", key);
    return vars_.at(key)->as<T>();
  }

  void startGroup(const std::string &name) { currentGroup_ = name; }
  void endGroup() { currentGroup_ = ""; }

  std::shared_ptr<CLI::App> app() { return app_; }

  YAML::Node getConfig() {
    // TODO: remove debugs
    YAML::Emitter emit;
    OutputYaml(config_, emit);
    std::cerr << emit.c_str() << std::endl;

    return config_;
  }
};

}  // namespace cli
}  // namespace marian
