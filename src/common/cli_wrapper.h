#pragma once

#include "3rd_party/CLI/CLI.hpp"
#include "3rd_party/any_type.h"
#include "3rd_party/yaml-cpp/yaml.h"
#include "common/definitions.h"

#include <iostream>
#include <map>
#include <string>
#include <unordered_set>
#include <vector>

namespace marian {

class Options;

namespace cli {

// Try to determine the width of the terminal
//
// TODO: make use of it in the current CLI or remove. This is an old code used
// for boost::program_options and might not be needed anymore.
//static uint16_t guess_terminal_width(uint16_t max_width = 0,
//                                     uint16_t default_width = 180);

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
  virtual std::string make_option_desc(const CLI::Option *) const override;

private:
  size_t screenWidth_{0};
};

// @TODO: in this file review the use of naked pointers. We use Ptr<Type> anywhere else,
// what's up with that?

/**
 * The helper structure storing an option object, the associated variable and creation index.
 */
struct CLIOptionTuple {
  CLI::Option *opt;
  Ptr<any_type> var;
  size_t idx{0};
  bool modified{false};
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
  // Map with option names and option tuples
  std::unordered_map<std::string, CLIOptionTuple> options_;
  // Counter for created options
  size_t counter_{0};
  // Command-line argument parser
  Ptr<CLI::App> app_;

  // Name of the default option group
  std::string defaultGroup_{""};
  // Name of the current option group
  std::string currentGroup_{""};

  // Reference to the main config object
  YAML::Node &config_;

  // Option for --version flag. This is a special flag and similarly to --help,
  // the key "version" will be not added into the YAML config
  CLI::Option *optVersion_;

  static std::string failureMessage(const CLI::App *app, const CLI::Error &e);

  // Extract option name from a comma-separated list of long and short options, e.g. 'help' from
  // '--help,-h'
  std::string keyName(const std::string &args) const {
    // re-use existing functions from CLI11 to keep option names consistent
    return std::get<1>(
               CLI::detail::get_names(CLI::detail::split_names(args)))  // get long names only
        .front();                                                       // get first long name
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
             size_t columnWidth = 40,
             size_t screenWidth = 0);

  /**
   * @brief Create an instance of the command-line argument parser,
   * short-cuft for Options object.
   *
   * @see Other constructor
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
   * Explicit default values will appear in help messages.
   *
   * @param args Comma-separated list of short and long option names
   * @param help Help message
   * @param val Default value
   *
   * @return Option object
   */
  template <typename T>
  CLI::Option *add(const std::string &args, const std::string &help, T val) {
    return addOption<T>(keyName(args),
                        args,
                        help,
                        val,
                        /*defaulted =*/true);
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
   * Implicit default values will *NOT* appear in help messages.
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
    return addOption<T>(keyName(args),
                        args,
                        help,
                        T(),
                        /*defaulted =*/false);
  }

  /**
   * Switch to different option group or to the default group if argument is empty.
   *
   * @param name Header of the option group
   */
  void switchGroup(const std::string &name = "");

  // Parse command-line arguments. Handles --help and --version options
  void parse(int argc, char **argv);

  /*
   * @brief Overwrite values for unparsed options
   *
   * Default values are overwritten with the options from the config provided, while parsed
   * command-line options remain unchanged.
   * This should be a preferred way of updating config options as the class keeps track of options,
   * which values have changed.
   *
   * @param config YAML config with new default values for options
   * @param errorMsg error message printed if config contains undefined keys. The message is
   *   appended with ": * <comma-separated list of invalid options>"
   */
  void updateConfig(const YAML::Node &config, const std::string &errorMsg);

  // Get textual YAML representation of the config
  std::string dumpConfig(bool skipDefault = false) const;

private:
  // Get names of options passed via command-line
  std::unordered_set<std::string> getParsedOptionNames() const;
  // Get option names in the same order as they are created
  std::vector<std::string> getOrderedOptionNames() const;

  template <typename T,
            // options with numeric and string-like values
            CLI::enable_if_t<!CLI::is_bool<T>::value && !CLI::is_vector<T>::value,
                             CLI::detail::enabler> = CLI::detail::dummy>
  CLI::Option *addOption(const std::string &key,
                         const std::string &args,
                         const std::string &help,
                         T val,
                         bool defaulted) {
    // add key to YAML
    config_[key] = val;

    // create option tuple
    CLIOptionTuple option;
    option.idx = counter_++;
    option.var = std::make_shared<any_type>(val);

    // callback function collecting a command-line argument
    CLI::callback_t fun = [this, key](CLI::results_t res) {
      options_[key].modified = true;
      // get variable associated with the option
      auto &var = options_[key].var->as<T>();
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

    // store option tuple
    option.opt = opt;
    options_.insert(std::make_pair(key, option));
    return options_[key].opt;
  }

  template <typename T,
            // options with vector values
            CLI::enable_if_t<CLI::is_vector<T>::value, CLI::detail::enabler> = CLI::detail::dummy>
  CLI::Option *addOption(const std::string &key,
                         const std::string &args,
                         const std::string &help,
                         T val,
                         bool defaulted) {
    // add key to YAML
    config_[key] = val;

    // create option tuple
    CLIOptionTuple option;
    option.idx = counter_++;
    option.var = std::make_shared<any_type>(val);

    // callback function collecting command-line arguments
    CLI::callback_t fun = [this, key](CLI::results_t res) {
      options_[key].modified = true;
      // get vector variable associated with the option
      auto &vec = options_[key].var->as<T>();
      vec.clear();
      bool ret = true;
      // handle '[]' as an empty vector
      if(res.size() == 1 && res.front() == "[]") {
        ret = true;
      } else {
        // populate the vector with parser results
        for(const auto &a : res) {
          vec.emplace_back();
          ret &= CLI::detail::lexical_cast(a, vec.back());
        }
        ret &= !vec.empty();
      }
      // update YAML entry
      config_[key] = vec;
      return ret;
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

    // store option tuple
    option.opt = opt;
    options_.insert(std::make_pair(key, option));
    return options_[key].opt;
  }

  template <typename T,
            // options with boolean values, called flags in CLI11
            CLI::enable_if_t<CLI::is_bool<T>::value, CLI::detail::enabler> = CLI::detail::dummy>
  CLI::Option *addOption(const std::string &key,
                         const std::string &args,
                         const std::string &help,
                         T val,
                         bool defaulted) {
    // add key to YAML
    config_[key] = val;

    // create option tuple
    CLIOptionTuple option;
    option.idx = counter_++;
    option.var = std::make_shared<any_type>(val);

    // callback function setting the flag
    CLI::callback_t fun = [this, key](CLI::results_t res) {
      options_[key].modified = true;
      // get parser result, it is safe as boolean options have an implicit value
      auto val = res[0];
      auto ret = true;
      if(val == "true" || val == "on" || val == "yes" || val == "1") {
        options_[key].var->as<T>() = true;
        config_[key] = true;
      } else if(val == "false" || val == "off" || val == "no" || val == "0") {
        options_[key].var->as<T>() = false;
        config_[key] = false;
      } else {
        ret = false;
      }
      return ret;
    };

    auto opt = app_->add_option(args, fun, help, defaulted);
    // set option group
    if(!currentGroup_.empty())
      opt->group(currentGroup_);
    // set textual representation of the default value for help message
    if(defaulted)
      opt->default_str(val ? "true" : "false");
    // allow to use the flag without any argument
    opt->implicit_val("true");

    // store option tuple
    option.opt = opt;
    options_.insert(std::make_pair(key, option));
    return options_[key].opt;
  }
};

}  // namespace cli
}  // namespace marian
