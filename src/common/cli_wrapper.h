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

// Option priority
enum struct OptionPriority : int { DefaultValue = 0, ConfigFile = 1, CommandLine = 2 };

/**
 * Helper tuple storing an option object, the associated variable and creation index
 *
 * Note: bare pointers are used for CLI::Option objects as this comes from the CLI11 library.
 * Removing it would require deep modifications in the 3rd party library, what we want to avoid.
 */
struct CLIOptionTuple {
  CLI::Option *opt;     // a pointer to an option object from CLI11
  Ptr<any_type> var;    // value assigned to the option via command-line
  size_t idx{0};        // order in which the option was created
  OptionPriority priority{cli::OptionPriority::DefaultValue};
};

// Helper tuple for aliases storing the alias name, value, and options to be expanded
struct CLIAliasTuple {
  std::string key;    // alias option name
  std::string value;  // value for the alias option indicating that it should be expanded
  YAML::Node config;  // config with options that the alias adds
};

// The helper class for cli::CLIWrapper handling formatting of options and their descriptions.
class CLIFormatter : public CLI::Formatter {
public:
  CLIFormatter(size_t columnWidth, size_t screenWidth);
  virtual std::string make_option_desc(const CLI::Option*) const override;

private:
  size_t screenWidth_{0};
};

/**
 * @brief The class used to define and parse command-line arguments.
 *
 * It is a wrapper around https://github.com/CLIUtils/CLI11 that stores defined command-line
 * arguments in a YAML object.
 *
 * Usage outline: first call add() methods to create all the options; then call parse(argv, argc) to
 * parse command line and get defined options and their values in a YAML object; finally call
 * parseAliases() to expand alias options. The config object can be also obtained later by calling
 * getConfig().
 *
 * Options are organized in option groups. Each option group has a header that preceeds all options
 * in the group. The header for the default option group can be set from the class constructor.
 */
class CLIWrapper {
private:
  // Map with option names and option tuples
  std::unordered_map<std::string, CLIOptionTuple> options_;
  // Counter for created options to keep track of order in which options were created
  size_t counter_{0};
  std::vector<CLIAliasTuple> aliases_;  // List of alias tuples

  Ptr<CLI::App> app_;                   // Command-line argument parser from CLI11

  std::string defaultGroup_{""};        // Name of the default option group
  std::string currentGroup_{""};        // Name of the current option group

  YAML::Node &config_;                  // Reference to the main config object

  // Option for --version flag. This is a special flag and similarly to --help,
  // the key "version" will be not added into the YAML config
  CLI::Option *optVersion_;

  // Extract option name from a comma-separated list of long and short options, e.g. 'help' from
  // '--help,-h'
  std::string keyName(const std::string &args) const;

  // Get names of options passed via command-line
  std::unordered_set<std::string> getParsedOptionNames() const;
  // Get option names in the same order as they are created
  std::vector<std::string> getOrderedOptionNames() const;

  static std::string failureMessage(const CLI::App *app, const CLI::Error &e);

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
   * @param screenWidth Maximum allowed width for help messages, 0 means no limit
   */
  CLIWrapper(YAML::Node &config,
             const std::string &description = "",
             const std::string &header = "General options",
             const std::string &footer = "",
             size_t columnWidth = 40,
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
  CLI::Option* add(const std::string& args, const std::string& help, T val) {
    return addOption<T>(keyName(args),
                        args,
                        help,
                        val,
                        /*defaulted =*/true);
  }

  /**
   * @brief Define an option without an explicit default value. The implicit default value is T()
   *
   * The option will be defined in the config file even if not given as a command-line argument. The
   * implicit default value for a boolean or numeric option is 0, for a string is an empty string,
   * and for a vector is an empty vector.
   *
   * Implicit default values will *NOT* appear in help messages.
   *
   * @param args Comma-separated list of short and long option names
   * @param help Help message
   *
   * @return Option object
   *
   * @TODO: require to always state the default value creating the parser as this will be clearer
   */
  template <typename T>
  CLI::Option* add(const std::string& args, const std::string& help) {
    return addOption<T>(keyName(args),
                        args,
                        help,
                        T(),
                        /*defaulted =*/false);
  }

  /**
   * @brief Transform a command line option into an alias. This alias will set other options later.
   *
   * An alias sets one or more options to predefined values. The options expanded by the alias are
   * provided as a function setting a temporary YAML config.
   *
   * The alias option has to be first defined using `add<T>()`. Otherwise, the program will abort.
   *
   * Defining more than one alias for the same `key` but different `value` is allowed.
   *
   * Option values are compared as std::string. If the alias option is a vector, the alias will be
   * triggered if `value` exists in that vector at least once.
   *
   * Options set directly via command line have precedence over options defined in an alias, i.e. an
   * option added via alias can be overwritten by setting a specific option via command line.
   *
   * @param key Alias option name
   * @param value Option value that trigger the alias
   * @param fun Function setting a temporary YAML config with options expanded by alias
   */
  void alias(const std::string &key,
             const std::string &value,
             const std::function<void(YAML::Node &config)> &fun) {
    ABORT_IF(!options_.count(key), "Option '{}' is not defined so alias can not be created", key);
    aliases_.resize(aliases_.size() + 1);
    aliases_.back().key = key;
    aliases_.back().value = value;
    fun(aliases_.back().config);
  }

  /**
   * Switch to different option group or to the default group if argument is empty.
   *
   * @param name Header of the option group
   * @return Previous group.
   */
  std::string switchGroup(std::string name = "");

  // Parse command-line arguments. Handles --help and --version options
  void parse(int argc, char** argv);

  /**
   * @brief Expand aliases based on arguments parsed with parse(int, char**)
   *
   * Should be called after parse(int, char**) to take an effect.  If any alias tries to expand an
   * undefined option, the method will abort the program.
   *
   * All options defined as aliases are removed from the global config object to avoid redundancy
   * when options are dumped (explicitly or implicitly) to a config file.
   */
  void parseAliases();

  /**
   * @brief Overwrite options with lower priority
   *
   * Values for options with lower priority than the provided priority remain unchanged. This allows
   * for overwritting default options by options from config files, or both by options provided in
   * the command line.
   *
   * This should be a preferred way of updating config options as the class keeps track of options,
   * which values have changed.
   *
   * @param config YAML config with new default values for options
   * @param priority priority of incoming options
   * @param errorMsg error message printed if config contains undefined keys. The message is
   *   appended with ": <comma-separated list of invalid options>"
   */
  void updateConfig(const YAML::Node &config, cli::OptionPriority priority, const std::string &errorMsg);

  // Get textual YAML representation of the config
  std::string dumpConfig(bool skipUnmodified = false) const;

private:
  template <typename T>
  using EnableIfNumbericOrString = CLI::enable_if_t<!CLI::is_bool<T>::value
                                   && !CLI::is_vector<T>::value, CLI::detail::enabler>;

  template <typename T, EnableIfNumbericOrString<T> = CLI::detail::dummy>
  CLI::Option* addOption(const std::string &key,
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
      options_[key].priority = cli::OptionPriority::CommandLine;
      // get variable associated with the option
      auto& var = options_[key].var->as<T>();
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

  template <typename T>
  using EnableIfVector = CLI::enable_if_t<CLI::is_vector<T>::value, CLI::detail::enabler>;

  template <typename T, EnableIfVector<T> = CLI::detail::dummy>
  CLI::Option* addOption(const std::string &key,
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
      options_[key].priority = cli::OptionPriority::CommandLine;
      // get vector variable associated with the option
      auto& vec = options_[key].var->as<T>();
      vec.clear();
      bool ret = true;
      // handle '[]' as an empty vector
      if(res.size() == 1 && res.front() == "[]") {
        ret = true;
      } else {
        // populate the vector with parser results
        for(const auto& a : res) {
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

  template <typename T>
  using EnableIfBoolean = CLI::enable_if_t<CLI::is_bool<T>::value, CLI::detail::enabler>;

  template <typename T, EnableIfBoolean<T> = CLI::detail::dummy>
  CLI::Option* addOption(const std::string &key,
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
      options_[key].priority = cli::OptionPriority::CommandLine;
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
