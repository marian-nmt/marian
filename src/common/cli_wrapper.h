#pragma once

#include "3rd_party/CLI/CLI.hpp"
#include "3rd_party/yaml-cpp/yaml.h"
#include "common/some_type.h"

#include <map>
#include <string>
#include <iostream>

namespace marian {
namespace cli {

template <typename S, typename T>
S &operator<<(S &s, const std::vector<T> &v) {
  for(auto &x : v)
    s << x << " ";
  return s;
}

class CLIWrapper {
private:
  // Stores option variables
  std::map<std::string, std::shared_ptr<some_type>> vars_;
  // Stores option objects
  std::map<std::string, std::shared_ptr<CLI::Option>> opts_;
  // Command-line arguments parser
  CLI::App app_;

  YAML::Node config_;

public:
  CLIWrapper() {}
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

    vars_.insert(std::make_pair(key, std::make_shared<some_type>(val)));

    CLI::callback_t fun = [this, key](CLI::results_t res) {
      auto& var = vars_[key]->as<T>();
      auto ret = CLI::detail::lexical_cast(res[0], var);
      config_[key] = var;
      return ret;
    };

    std::shared_ptr<CLI::Option> opt(app_.add_option(args, fun, help, true));
    opt->type_name(CLI::detail::type_name<T>());
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
    std::cerr << "CLI::add(" << key << ") " << std::endl;
    config_[key] = false;
    vars_.insert(std::make_pair(key, std::make_shared<some_type>(false)));
    CLI::callback_t fun = [this, key](CLI::results_t res) {
      config_[key] = !res.empty();
      return true;
    };

    std::shared_ptr<CLI::Option> opt(app_.add_option(args, fun, help, true));
    opt->type_size(0);
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
    std::cerr << "CLI::add(" << key << ") " << std::endl;

    vars_.insert(std::make_pair(key, std::make_shared<some_type>(val)));

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

    std::shared_ptr<CLI::Option> opt(app_.add_option(args, fun, help));
    opt->type_name(CLI::detail::type_name<T>())->type_size(-1);
    opts_.insert(std::make_pair(key, opt));
    return opts_[key];
  }


  template <typename T>
  std::shared_ptr<CLI::Option> getOption(const std::string &key) {
    std::cerr << "CLI::getOption(" << key << ") .count=" << opts_[key]->count()
              << std::endl;
    return opts_[key];
  }

  template <typename T>
  T get(const std::string &key) {
    std::cerr << "CLI::get(" << key << ") =" << vars_[key]->as<T>()
              << " .count=" << opts_[key]->count()
              << " .bool=" << (bool)(*opts_[key])
              << " .empty=" << opts_[key]->empty() << std::endl;
    return vars_[key]->as<T>();
  }

  bool parse(int argv, char **argc) { app_.parse(argv, argc); }

  CLI::App *app() { return &app_; }

  YAML::Node getConfig() { return config_; }
};

}  // namespace cli
}  // namespace marian
