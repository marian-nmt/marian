#include "common/cli_wrapper.h"
#include "common/cli_helper.h"
#include "common/logging.h"
#include "common/options.h"
#include "common/timer.h"
#include "common/utils.h"
#include "common/version.h"

namespace marian {
namespace cli {

// clang-format off
const std::unordered_set<std::string> DEPRECATED_OPTIONS = {
  "version",
  "special-vocab",
// @TODO: uncomment once we actually deprecate them.
//  "after-batches",
//  "after-epochs"
};
// clang-format on


/*
static uint16_t guess_terminal_width(uint16_t max_width, uint16_t default_width) {
  uint16_t cols = 0;
#ifdef TIOCGSIZE
  struct ttysize ts;
  ioctl(STDIN_FILENO, TIOCGSIZE, &ts);
  if(ts.ts_cols != 0)
    cols = ts.ts_cols;
#elif defined(TIOCGWINSZ)
  struct winsize ts;
  ioctl(STDIN_FILENO, TIOCGWINSZ, &ts);
  if(ts.ws_col != 0)
    cols = ts.ws_col;
#endif
  // couldn't determine terminal width
  if(cols == 0)
    cols = default_width;
  return max_width ? std::min(cols, max_width) : cols;
}
*/

CLIFormatter::CLIFormatter(size_t columnWidth, size_t screenWidth)
    : CLI::Formatter(), screenWidth_(screenWidth) {
  column_width(columnWidth);
}

std::string CLIFormatter::make_option_desc(const CLI::Option *opt) const {
  auto desc = opt->get_description();

  // TODO: restore guessing terminal width

  // wrap lines in the option description
  if(screenWidth_ > 0 && screenWidth_ < desc.size() + get_column_width()) {
    size_t maxWidth = screenWidth_ - get_column_width();
    std::istringstream descIn(desc);
    std::ostringstream descOut;

    size_t len = 0;
    std::string word;
    while(descIn >> word) {
      if(len > 0)
        descOut << " ";
      if(len + word.length() > maxWidth) {
        descOut << '\n' << std::string(get_column_width(), ' ');
        len = 0;
      }
      descOut << word;
      len += word.length() + 1;
    }

    desc = descOut.str();
  }
  return desc;
}

CLIWrapper::CLIWrapper(YAML::Node &config,
                       const std::string &description,
                       const std::string &header,
                       const std::string &footer,
                       size_t columnWidth,
                       size_t screenWidth)
    : app_(std::make_shared<CLI::App>(description)),
      defaultGroup_(header),
      currentGroup_(header),
      config_(config) {
  // set footer
  if(!footer.empty())
    app_->footer("\n" + footer);

  // set group name for the automatically added --help option
  app_->get_help_ptr()->group(defaultGroup_);

  // set custom failure message
  app_->failure_message(failureMessage);
  // set custom formatter for help message
  auto fmt = std::make_shared<CLIFormatter>(columnWidth, screenWidth);
  app_->formatter(fmt);

  // add --version option
  optVersion_ = app_->add_flag("--version", "Print the version number and exit");
  optVersion_->group(defaultGroup_);
}

CLIWrapper::~CLIWrapper() {}

// set current group to name, return previous group
std::string CLIWrapper::switchGroup(std::string name) {
  currentGroup_.swap(name);
  if (currentGroup_.empty())
    currentGroup_ = defaultGroup_;
  return name;
}

void CLIWrapper::parse(int argc, char** argv) {
  try {
    app_->parse(argc, argv);
  } catch(const CLI::ParseError& e) {
    exit(app_->exit(e));
  }

  // handle --version flag
  if(optVersion_->count()) {
    std::cerr << buildVersion() << std::endl;
    exit(0);
  }
}

void CLIWrapper::parseAliases() {
  // Exit if no aliases defined
  if(aliases_.empty())
    return;

  // Find the set of values allowed for each alias option.
  // Later we will check and abort if an alias option has an unknown value.
  std::unordered_map<std::string, std::unordered_set<std::string>> allowedAliasValues;
  for(auto &&alias : aliases_)
    allowedAliasValues[alias.key].insert(alias.value);

  // Iterate all known aliases, each alias has a key, value, and config
  for(auto &&alias : aliases_) {
    // Check if the alias option exists in the config (it may come from command line or a config
    // file)
    if(config_[alias.key]) {
      // Check if the option in the config stores the value required to expand the alias. If so,
      // expand the alias.
      // Two cases:
      //  * the option is a sequence: extract it as a vector of strings and look for the value
      //  * otherwise: compare values as strings
      bool expand = false;
      if(config_[alias.key].IsSequence()) {
        auto aliasOpts = config_[alias.key].as<std::vector<std::string>>();
        // Abort if an alias option has an unknown value, i.e. value that has not been defined
        // in common/aliases.cpp
        for(auto &&aliasOpt : aliasOpts)
          if(allowedAliasValues[alias.key].count(aliasOpt) == 0) {
            std::vector<std::string> allowedOpts(allowedAliasValues[alias.key].begin(),
                                                 allowedAliasValues[alias.key].end());
            ABORT("Unknown value '" + aliasOpt + "' for alias option --" + alias.key + ". "
                  "Allowed values: " + utils::join(allowedOpts, ", "));
          }
        expand = std::find(aliasOpts.begin(), aliasOpts.end(), alias.value) != aliasOpts.end();
      } else {
        expand = config_[alias.key].as<std::string>() == alias.value;
      }

      if(expand) {
        // Update global config options with the config associated with the alias. Abort if the
        // alias contains an undefined option.
        updateConfig(alias.config,
                     // Priority of each expanded option is the same as the priority of the alias
                     options_[alias.key].priority,
                     "Unknown option(s) in alias '" + alias.key + ": " + alias.value + "'");
      }
    }
  }

  // Remove aliases from the global config to avoid redundancy when writing/reading config files
  for(const auto &alias : aliases_) {
    config_.remove(alias.key);
  }
}

std::string CLIWrapper::keyName(const std::string& args) const {
  // re-use existing functions from CLI11 to keep option names consistent
  return std::get<1>(
              CLI::detail::get_names(CLI::detail::split_names(args)))  // get long names only
      .front();                                                        // get first long name
}

void CLIWrapper::updateConfig(const YAML::Node &config, cli::OptionPriority priority, const std::string &errorMsg) {
  auto cmdOptions = getParsedOptionNames();
  // Keep track of unrecognized options from the provided config
  std::vector<std::string> unknownOpts;

  // Iterate incoming options: they need to be merged into the global config
  for(auto it : config) {
    auto key = it.first.as<std::string>();

    // Skip options specified via command-line to allow overwriting them
    if(cmdOptions.count(key))
      continue;
    // Skip options that might exist in config files generated by older versions of Marian
    if(DEPRECATED_OPTIONS.count(key))
      continue;

    // Check if an incoming option has been defined in CLI
    if(options_.count(key)) {
      // Do not proceed if the priority of incoming option is not greater than the existing option
      if(priority <= options_[key].priority) {
        continue;
      }
      // Check if the option exists in the global config and types match
      if(config_[key] && config_[key].Type() == it.second.Type()) {
        config_[key] = YAML::Clone(it.second);
        options_[key].priority = priority;
        // If types doesn't match, try to convert
      } else {
        // Default value is a sequence and incoming node is a scalar, hence we can upcast to
        // single element sequence
        if(config_[key].IsSequence() && it.second.IsScalar()) {
          // create single element sequence
          YAML::Node sequence;
          sequence.push_back(YAML::Clone(it.second));
          config_[key] = sequence;  // overwrite to replace default values
          options_[key].priority = priority;
        } else {
          // Cannot convert other non-matching types, e.g. scalar <- list should fail
          ABORT("Cannot convert values for the option: " + key);
        }
      }
    } else {  // an unknown option
      unknownOpts.push_back(key);
    }
  }

  ABORT_IF(!unknownOpts.empty(), errorMsg + ": " + utils::join(unknownOpts, ", "));
}

std::string CLIWrapper::dumpConfig(bool skipUnmodified /*= false*/) const {
  YAML::Emitter out;
  out << YAML::Comment("Marian configuration file generated at " + timer::currentDate()
                       + " with version " + buildVersion());
  out << YAML::BeginMap;
  std::string comment;
  // Iterate option names in the same order as they have been created
  for(const auto &key : getOrderedOptionNames()) {
    // Do not dump options that were removed from config_
    if(!config_[key])
      continue;
    // Do not dump options that were not passed via the command line
    if(skipUnmodified && options_.at(key).priority == cli::OptionPriority::DefaultValue)
      continue;
    // Put the group name as a comment before the first option in the group
    auto group = options_.at(key).opt->get_group();
    if(comment != group) {
      if(!comment.empty())
        out << YAML::Newline;
      comment = group;
      out << YAML::Comment(group);
    }
    out << YAML::Key;
    out << key;
    out << YAML::Value;
    cli::OutputYaml(config_[key], out);
  }
  out << YAML::EndMap;
  return out.c_str();
}

std::unordered_set<std::string> CLIWrapper::getParsedOptionNames() const {
  std::unordered_set<std::string> keys;
  for(const auto &it : options_)
    if(!it.second.opt->empty())
      keys.emplace(it.first);
  return keys;
}

std::vector<std::string> CLIWrapper::getOrderedOptionNames() const {
  std::vector<std::string> keys;
  // extract all option names
  for(auto const &it : options_)
    keys.push_back(it.first);
  // sort option names by creation index
  sort(keys.begin(), keys.end(), [this](const std::string& a, const std::string& b) {
    return options_.at(a).idx < options_.at(b).idx;
  });
  return keys;
}

std::string CLIWrapper::failureMessage(const CLI::App *app, const CLI::Error &e) {
  std::string header = "Error: " + std::string(e.what()) + "\n";
  if(app->get_help_ptr() != nullptr)
    header += "Run with " + app->get_help_ptr()->get_name() + " for more information.\n";
  return header;
}

}  // namespace cli
}  // namespace marian
