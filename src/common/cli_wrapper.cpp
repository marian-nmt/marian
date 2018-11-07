#include "common/cli_wrapper.h"
#include "common/options.h"
#include "common/version.h"

namespace marian {
namespace cli {

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
  optVersion_
      = app_->add_flag("--version", "Print the version number and exit");
  optVersion_->group(defaultGroup_);
}

CLIWrapper::CLIWrapper(Ptr<marian::Options> options,
                       const std::string &description,
                       const std::string &header,
                       const std::string &footer,
                       size_t columnWidth,
                       size_t screenWidth)
    : CLIWrapper(options->getYaml(),
                 description,
                 header,
                 footer,
                 columnWidth,
                 screenWidth) {}

CLIWrapper::~CLIWrapper() {}

void CLIWrapper::switchGroup(const std::string &name) {
  if(name.empty())
    currentGroup_ = defaultGroup_;
  else
    currentGroup_ = name;
}

void CLIWrapper::parse(int argc, char **argv) {
  try {
    app_->parse(argc, argv);
  } catch(const CLI::ParseError &e) {
    exit(app_->exit(e));
  }

  // handle --version flag
  if(optVersion_->count()) {
    std::cerr << buildVersion() << std::endl;
    exit(0);
  }
}

std::string CLIWrapper::failureMessage(const CLI::App *app,
                                       const CLI::Error &e) {
  std::string header = "Error: " + std::string(e.what()) + "\n";
  if(app->get_help_ptr() != nullptr)
    header += "Run with " + app->get_help_ptr()->get_name()
              + " for more information.\n";
  return header;
}

void CLIWrapper::overwriteDefault(const YAML::Node &node) {
  // iterate requested default values
  for(auto it : node) {
    auto key = it.first.as<std::string>();
    // warn if the option for which the default value we are setting for has
    // been not defined
    if(allVars_.count(key) == 0)
      LOG(warn, "Default value for an undefined option with key '{}'", key);
    // if we have an option but it was not specified on command-line
    if(allVars_.count(key) > 0 && opts_.at(key)->empty()) {
      config_[key] = YAML::Clone(it.second);
    }
  }
}

}  // namespace cli
}  // namespace marian
