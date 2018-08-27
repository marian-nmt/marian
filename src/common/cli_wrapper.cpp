#include "common/cli_wrapper.h"
#include "common/version.h"

namespace marian {
namespace cli {

CLIWrapper::CLIWrapper(const std::string &name)
    : app_(std::make_shared<CLI::App>()),
      defaultGroup_(name),
      currentGroup_(name) {
  app_->get_help_ptr()->group(defaultGroup_);
  app_->failure_message(failureMessage);
}

CLIWrapper::~CLIWrapper() {}

void CLIWrapper::parse(int argc, char** argv) {
  try {
    app_->parse(argc, argv);
  } catch(const CLI::ParseError& e) {
    exit(app_->exit(e));
  }

  if(has("version")) {
    std::cerr << PROJECT_VERSION_FULL << std::endl;
    exit(0);
  }
}

bool CLIWrapper::has(const std::string &key) const {
  return opts_.count(key) > 0 && !opts_.at(key)->empty();
}

std::string CLIWrapper::failureMessage(const CLI::App *app,
                                       const CLI::Error &e) {
  std::string header = "Error: " + std::string(e.what()) + "\n";
  if(app->get_help_ptr() != nullptr)
    header += "Run with " + app->get_help_ptr()->get_name()
              + " for more information.\n";
  return header;
}

YAML::Node CLIWrapper::getConfig() const {
  return config_;
}

YAML::Node CLIWrapper::getConfigWithNewDefaults(const YAML::Node &node) const {
  YAML::Node yaml = YAML::Clone(config_);
  // iterate requested default values
  for(auto it : node) {
    auto key = it.first.as<std::string>();
    // warn if the option for which the default value we are setting for has
    // been not defined
    if(vars_.count(key) == 0)
      LOG(warn, "Default value for an undefined option with key '{}'", key);
    // if we have an option and but it was not specified on command-line
    if(vars_.count(key) > 0 && opts_.at(key)->empty()) {
      yaml[key] = YAML::Clone(it.second);
    }
  }
  return yaml;
}

}  // namespace cli
}  // namespace marian
