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

public:
  CLIWrapper() {}
  virtual ~CLIWrapper() {}

  template <typename T>
  std::shared_ptr<CLI::Option> add(const std::string &key,
                                   const std::string &args,
                                   const std::string &help,
                                   T val = T()) {
    std::cerr << "CLI::add(" << key << ") " << std::endl;

    vars_.insert(std::make_pair(key, std::make_shared<some_type>(val)));
    std::shared_ptr<CLI::Option> opt(
        app_.add_option(args, vars_[key]->as<T>(), help));
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
};

template <>
std::shared_ptr<CLI::Option> CLIWrapper::add(const std::string &key,
                                             const std::string &args,
                                             const std::string &help,
                                             bool val) {
  std::cerr << "CLI::add(" << key << ") " << std::endl;

  vars_.insert(std::make_pair(key, std::make_shared<some_type>(false)));
  std::shared_ptr<CLI::Option> opt(app_.add_flag(args, help));
  opts_.insert(std::make_pair(key, opt));
  return opts_[key];
}

}  // namespace cli
}  // namespace marian
