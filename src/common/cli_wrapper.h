#pragma once

#include "3rd_party/CLI/CLI.hpp"
#include "3rd_party/yaml-cpp/yaml.h"
#include "common/definitions.h"
#include "common/logging.h"
#include "common/some_type.h"

#include <iostream>
#include <map>
#include <string>

namespace marian {
namespace cli {

// try to determine the width of the terminal
static uint16_t guess_terminal_width(uint16_t max_width = 0, uint16_t default_width = 180);

namespace validators {
const CLI::detail::ExistingFileValidator file_exists;
const CLI::detail::ExistingDirectoryValidator dir_exists;
const CLI::detail::ExistingPathValidator path_exists;

const CLI::detail::NonexistentPathValidator path_not_exists;

typedef CLI::Range range;
}

class CLIFormatter : public CLI::Formatter {
public:
  CLIFormatter(size_t columnWidth, size_t screenWidth);
  virtual std::string make_option_desc(const CLI::Option *) const;

private:
  size_t screenWidth_{0};
};

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

  // Name of the default option group
  std::string defaultGroup_{""};
  // Name of the current option group
  std::string currentGroup_{""};

  static std::string failureMessage(const CLI::App *app, const CLI::Error &e);

  std::string keyName(const std::string &args) const {
    return std::get<1>(CLI::detail::get_names(CLI::detail::split_names(args)))
        .front();
  }

public:
  /**
   * @brief Creates an instance of the command-line parser
   *
   * Option --help, -h is automatically added.
   *
   * @param name Header for the main option group
   * @param columnWidth Width of the column with option names
   * @param screenWidth Maximum allowed width for help messages
   */
  CLIWrapper(const std::string &name = "General options",
             size_t columnWidth = 35,
             size_t screenWidth = 0);

  virtual ~CLIWrapper();

  /**
   * @brief Defines an option with a default value
   *
   * @param args Comma-separated list of short and long option names
   * @param help Help message
   * @param val Default value
   *
   * @return Option object
   */
  template <typename T>
  CLI::Option *add(const std::string &args, const std::string &help, T val) {
    return add_option<T>(keyName(args), args, help, val, true, true);
  }

  /**
   * @brief Defines an option without an explicit default value. The implicit
   * default value is T()
   *
   * The option will be defined in the config file even if not given as a
   * command-line argument. The implicit default value for a numeric option is
   * 0, for a string is an empty string, and for a vector is an empty vector.
   *
   * @param args Comma-separated list of short and long option names
   * @param help Help message
   *
   * @return Option object
   */
  template <typename T>
  CLI::Option *add(const std::string &args, const std::string &help) {
    return add_option<T>(keyName(args), args, help, T(), false, true);
  }

  /**
   * @brief Defines a non-defaulted option
   *
   * The option will be not present in the config file unless given as a
   * command-line argument.
   *
   * @param args Comma-separated list of short and long option names
   * @param help Help message
   *
   * @return Option object
   */
  template <typename T>
  CLI::Option *add_nondefault(const std::string &args,
                              const std::string &help) {
    return add_option<T>(keyName(args), args, help, T(), false, false);
  }

  /**
   * @brief Switch to different option group or to the default group if
   * argument is empty
   *
   * @param name Header of the option group
   */
  void switchGroup(const std::string &name = "");

  // Parses command-line arguments. Handles --help and --version options
  void parse(int argc, char** argv);

  // Checks if an option has been defined (not necessarily parsed)
  bool has(const std::string &key) const;

  // Gets the current value for the option
  template <typename T>
  T get(const std::string &key) const {
    ABORT_IF(
        vars_.count(key) == 0, "An option with key '{}' is not defined", key);
    return vars_.at(key)->as<T>();
  }

  /**
   * @brief Return config with defined and parsed options
   *
   * @return YAML config
   */
  YAML::Node getConfig() const;

  /**
   * @brief Generate config with overwritten values for unparsed options
   *
   * Default values are overwritten with the options found in the config
   * provided as the argument and parsed command-line options remain unchanged
   *
   * @param node YAML config with new default values for options
   *
   * @return YAML config
   */
  YAML::Node getConfigWithNewDefaults(const YAML::Node& node) const;

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
    if(defaulted) {
      std::stringstream ss;
      ss << val;
      opt->default_str(ss.str());
    }

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
    if(defaulted)
      opt->default_str(CLI::detail::join(val));

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
};

}  // namespace cli
}  // namespace marian
