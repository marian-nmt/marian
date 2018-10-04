#pragma once

#include "3rd_party/CLI/CLI.hpp"
#include "3rd_party/any_type.h"
#include "3rd_party/yaml-cpp/yaml.h"
#include "common/definitions.h"
#include "common/logging.h"

#include <iostream>
#include <map>
#include <string>

namespace marian {

class Options;

namespace cli {

// Try to determine the width of the terminal
//
// TODO: make use of it in the current CLI or remove. This is an old code used
// for boost::program_options and might not be needed anymore.
static uint16_t guess_terminal_width(uint16_t max_width = 0,
                                     uint16_t default_width = 180);

// TODO: use validators in ConfigParser
namespace validators {
const CLI::detail::ExistingFileValidator file_exists;
const CLI::detail::ExistingDirectoryValidator dir_exists;
const CLI::detail::ExistingPathValidator path_exists;

const CLI::detail::NonexistentPathValidator path_not_exists;

typedef CLI::Range range;
}

/**
 * The helper class for cli::CLIWrapper handling formatting of options and their
 * descriptions.
 */
class CLIFormatter : public CLI::Formatter {
public:
  CLIFormatter(size_t columnWidth, size_t screenWidth);
  virtual std::string make_option_desc(const CLI::Option *) const;

private:
  size_t screenWidth_{0};
};

/**
 * @brief The class used to define and parse command-line arguments.
 *
 * It is a wrapper around https://github.com/CLIUtils/CLI11 that stores defined
 * command-line arguments in a YAML object.
 *
 * Usage outline: first call add() methods to create all the options; then call
 * parse(argv, argc) to parse command line and get defined options and their
 * values in a YAML object. The object can be also obtained later by calling
 * getConfig().
 *
 * Options are organized in option groups. Each option group has a header that
 * preceeds all options in the group. The header for the default option group
 * can be set from the class constructor.
 */
class CLIWrapper {
private:
  // [option name] -> option value
  std::map<std::string, Ptr<any_type>> allVars_;
  // Map with option names and objects
  std::map<std::string, CLI::Option *> opts_;
  // Command-line argument parser
  Ptr<CLI::App> app_;

  // Name of the default option group
  std::string defaultGroup_{""};
  // Name of the current option group
  std::string currentGroup_{""};

  // If this is a wrapper then this should just be a reference,
  // then we do not have the added level of containment.
  YAML::Node &config_;

  // Option for --version flag. This is a special flag and similarly to --help,
  // the key "version" will be not added into the YAML config
  CLI::Option* optVersion_;

  static std::string failureMessage(const CLI::App *app, const CLI::Error &e);

  // Extract an option name from comma-separated list of command-line arguments,
  // e.g. 'help' from '--help,-h'
  std::string keyName(const std::string &args) const {
    // re-use existing functions from CLI11 to keep option names consistent
    return std::get<1>(CLI::detail::get_names(CLI::detail::split_names(
                           args)))  // get long names only
        .front();                   // get first long name
  }

public:
  /**
   * @brief Create an instance of the command-line argument parser
   *
   * Option --help, -h is automatically added.
   *
   * @param config A reference to the to-be-wrapped yaml tree
   * @param description Program description
   * @param header Header text for the main option group
   * @param footer Text displayed after the list of options
   * @param columnWidth Width of the column with option names
   * @param screenWidth Maximum allowed width for help messages, 0 means no
   *  limit
   */
  CLIWrapper(YAML::Node &config,
             const std::string &description = "",
             const std::string &header = "General options",
             const std::string &footer = "",
             size_t columnWidth = 35,
             size_t screenWidth = 0);

  /**
   * @brief Create an instance of the command-line argument parser,
   * short-cuft for Options object.
   *
   * Option --help, -h is automatically added.
   *
   * @param options A smart pointer to the Options object containing the
   *  to-be-wrapped yaml tree
   * @param description Program description
   * @param header Header text for the main option group
   * @param footer Text displayed after the list of options
   * @param columnWidth Width of the column with option names
   * @param screenWidth Maximum allowed width for help messages, 0 means no
   *  limit
   */
  CLIWrapper(Ptr<Options> options,
             const std::string &description = "",
             const std::string &header = "General options",
             const std::string &footer = "",
             size_t columnWidth = 30,
             size_t screenWidth = 0);

  virtual ~CLIWrapper();

  /**
   * @brief Define an option with a default value
   *
   * @param args Comma-separated list of short and long option names
   * @param help Help message
   * @param val Default value
   *
   * @return Option object
   */
  template <typename T>
  CLI::Option *add(const std::string &args, const std::string &help, T val) {
    return add_option<T>(keyName(args),
                         args,
                         help,
                         val,
                         /*defaulted =*/true,
                         /*addToConfig =*/true);
  }

  /**
   * @brief Define an option without an explicit default value. The implicit
   * default value is T()
   *
   * The option will be defined in the config file even if not given as a
   * command-line argument. The implicit default value for a boolean or numeric
   * option is 0, for a string is an empty string, and for a vector is an empty
   * vector.
   *
   * @param args Comma-separated list of short and long option names
   * @param help Help message
   *
   * @return Option object
   *
   * TODO: require to always state the default value creating the parser as this
   * will be clearer
   */
  template <typename T>
  CLI::Option *add(const std::string &args, const std::string &help) {
    return add_option<T>(keyName(args),
                         args,
                         help,
                         T(),
                         /*defaulted =*/false,
                         /*addToConfig =*/true);
  }

  /**
   * @brief Define a non-defaulted option
   *
   * The option will not be present in the config file unless given as a
   * command-line argument.
   *
   * @param args Comma-separated list of short and long option names
   * @param help Help message
   *
   * @return Option object
   *
   * @TODO: consider removing this method during final refactorization of
   * command-line/config parsers in the future as all options should either
   * have a default value or be non-defaulted
   */
  template <typename T>
  CLI::Option *add_nondefault(const std::string &args,
                              const std::string &help) {
    return add_option<T>(keyName(args),
                         args,
                         help,
                         T(),
                         /*defaulted =*/false,
                         /*addToConfig =*/false);
  }

  /**
   * Switch to different option group or to the default group if
   * argument is empty.
   *
   * @param name Header of the option group
   */
  void switchGroup(const std::string &name = "");

  // Parse command-line arguments. Handles --help and --version options
  void parse(int argc, char **argv);

  /**
   * @brief Overwrite values for unparsed options
   *
   * Default values are overwritten with the options found in the config
   * provided as the argument, while parsed command-line options remain
   * unchanged
   *
   * @param node YAML config with new default values for options
   */
  void overwriteDefault(const YAML::Node &node);

private:
  template <
      typename T,
      // options with numeric and string-like values
      CLI::enable_if_t<!CLI::is_bool<T>::value && !CLI::is_vector<T>::value,
                       CLI::detail::enabler> = CLI::detail::dummy>
  CLI::Option *add_option(const std::string &key,
                          const std::string &args,
                          const std::string &help,
                          T val,
                          bool defaulted,
                          bool addToConfig) {
    // define YAML entry if requested
    if(addToConfig)
      config_[key] = val;
    // create variable for the option
    allVars_.insert(std::make_pair(key, std::make_shared<any_type>(val)));

    // callback function collecting a command-line argument
    CLI::callback_t fun = [this, key](CLI::results_t res) {
      // get variable associated with the option
      auto &var = allVars_[key]->as<T>();
      // store parser result in var
      auto ret = CLI::detail::lexical_cast(res[0], var);
      // update YAML entry
      config_[key] = var;
      return ret;
    };

    auto opt = app_->add_option(args, fun, help, defaulted);
    // set human readable type value: UINT, INT, FLOAT or TEXT
    opt->type_name(CLI::detail::type_name<T>());
    // set option group
    if(!currentGroup_.empty())
      opt->group(currentGroup_);
    // set textual representation of the default value for help message
    if(defaulted) {
      std::stringstream ss;
      ss << val;
      opt->default_str(ss.str());
    }

    // store option object
    opts_.insert(std::make_pair(key, opt));
    return opts_[key];
  }

  template <typename T,
            // options with vector values
            CLI::enable_if_t<CLI::is_vector<T>::value,
                             CLI::detail::enabler> = CLI::detail::dummy>
  CLI::Option *add_option(const std::string &key,
                          const std::string &args,
                          const std::string &help,
                          T val,
                          bool defaulted,
                          bool addToConfig) {
    // define YAML entry if requested
    if(addToConfig)
      config_[key] = val;
    // create variable for the option
    allVars_.insert(std::make_pair(key, std::make_shared<any_type>(val)));

    // callback function collecting command-line arguments
    CLI::callback_t fun = [this, key](CLI::results_t res) {
      // get vector variable associated with the option
      auto &vec = allVars_[key]->as<T>();
      vec.clear();
      bool ret = true;
      // populate the vector with parser results
      for(const auto &a : res) {
        vec.emplace_back();
        ret &= CLI::detail::lexical_cast(a, vec.back());
      }
      // update YAML entry
      config_[key] = vec;
      return (!vec.empty()) && ret;
    };

    auto opt = app_->add_option(args, fun, help);
    // set human readable type value: VECTOR and
    opt->type_name(CLI::detail::type_name<T>());
    // accept unlimited number of arguments
    opt->type_size(-1);
    // set option group
    if(!currentGroup_.empty())
      opt->group(currentGroup_);
    // set textual representation of the default vector values for help message
    if(defaulted)
      opt->default_str(CLI::detail::join(val));

    // store option object
    opts_.insert(std::make_pair(key, opt));
    return opts_[key];
  }

  template <typename T,
            // options with boolean values, called flags in CLI11
            CLI::enable_if_t<CLI::is_bool<T>::value,
                             CLI::detail::enabler> = CLI::detail::dummy>
  CLI::Option *add_option(const std::string &key,
                          const std::string &args,
                          const std::string &help,
                          T val,
                          bool defaulted,
                          bool addToConfig) {
    // define YAML entry if requested
    if(addToConfig)
      config_[key] = val;
    // create variable for the option
    allVars_.insert(std::make_pair(key, std::make_shared<any_type>(val)));

    // callback function setting the flag
    CLI::callback_t fun = [this, key](CLI::results_t res) {
      // set boolean variable associated with the option
      allVars_[key]->as<T>() = !res.empty();
      // update YAML entry
      config_[key] = !res.empty();
      return true;
    };

    auto opt = app_->add_option(args, fun, help, defaulted);
    // do not accept any argument for the boolean option
    opt->type_size(0);
    // set option group
    if(!currentGroup_.empty())
      opt->group(currentGroup_);

    // store option object
    opts_.insert(std::make_pair(key, opt));
    return opts_[key];
  }
};

}  // namespace cli
}  // namespace marian
